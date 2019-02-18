from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import cambrian
from models import DistilledUNetModel
from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment("distillation")

@ex.config
def distillation_config():
    args = {
        "a_input": [],
        "b_input": [],
        "a_channels": [3],
        "b_channels": [3],
        "a_eval": [],
        "b_eval": [],
        "a_temporals": [],

        "mode": "train",
        "model_dir": "models",
        "export_dir": "export",

        "epochs": 3000,
        
        "batch_size": 32,
        "ngf": 8,
        "ndf": 32,
        "init_stddev": 0.02,

        "crop_size": 256,
        "scale_size": 0,
        
        "lr_g": 0.0002,
        "lr_d": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,

        "metric_loss": "bce",
        "metric_weight": 100.0,
        
        "num_gpus": 1,

        "separable_conv": False,
        "no_disc_bn": False,
        "no_gen_bn": False,
        "layer_norm": False,
        "angle_output": False,
        "out_channels": 1,
    }

def get_specs_from_args(args, a_input_key, b_input_key):
    # If we only have single elements for inputs or channels
    # make a list out of them
    def ensure_list(x):
        if not isinstance(x, list) and not isinstance(x, tuple):
            return [x]
        return x
    a_input, b_input = ensure_list(args[a_input_key]), ensure_list(args[b_input_key])
    a_channels, b_channels = ensure_list(args["a_channels"]), ensure_list(args["b_channels"])

    num_a, num_b = len(a_input), len(b_input)

    # If only one channel was passed use that channel for all inputs
    if len(a_channels) == 1:
        a_channels = a_channels * num_a

    if len(b_channels) == 1:
        b_channels = b_channels * num_b

    # Other args
    scale_size = args["scale_size"]
    crop_size = args["crop_size"]

    if scale_size <= 0:
        scale_size = crop_size

    # Create the specs
    def _make_specs(inputs, channels):
        return [cambrian.nn.IOSpecification(index, start_channel, match_path, chans, scale_size, crop_size)
                for index, (start_channel, (match_path, chans))
                in enumerate(cambrian.utils.count_up(zip(inputs, channels), lambda mc: mc[1]))]

    a_specs = _make_specs(a_input, a_channels)
    b_specs = _make_specs(b_input, b_channels)

    return a_specs, b_specs

def get_parse_image_ab_fn(input_specs, output_specs, temporal_inputs=[]):
    def parse_image_ab(*file_names):
        num_inputs = len(input_specs)
        num_outputs = len(output_specs)
        assert len(file_names) == num_inputs + num_outputs

        def _parse(file_name, spec):
            image_data = tf.read_file(file_name)
            image = tf.image.convert_image_dtype(tf.image.decode_png(image_data, channels=spec.channels), spec.dtype)
            image = tf.image.resize_images(image, [spec.scale_size, spec.scale_size], method=tf.image.ResizeMethod.AREA) #reduces artifacts, consider as part of specs
            return image
            
        specs = input_specs + output_specs

        images = [_parse(file_name, spec) for file_name, spec in zip(file_names, specs)]

        input_images = images[:num_inputs]
        output_images = images[num_inputs:]
        
        input_dict = {cambrian.nn.get_input_name(i): img for i, img in enumerate(input_images)}
        output_dict = {cambrian.nn.get_output_name(i): img for i, img in enumerate(output_images)}

        # Add temporally warped inputs
        def _warp_temporally(image):
            # [a0, a1, a2, b0, b1, b2, c0, c1]
            # (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)
            # k = c0 x + c1 y + 1
            warp_params = [
                tf.random.uniform((2,), 0.9, 1.1, tf.float32),
                tf.random.uniform((2,), -50, 50, tf.float32),
                tf.random.uniform((2,), 0.9, 1.1, tf.float32),
                tf.random.uniform((2,), -50, 50, tf.float32),
                tf.random.uniform((2,), 0.9, 1.1, tf.float32),
            ]

            warp_fn = lambda im: tf.contrib.image.transform(image, warp_params, interpolation="BILINEAR")

            # 10% chance to have a blank image (ie. first frame)
            p_empty = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            empty = tf.less(p_empty, 0.1)
            result = tf.cond(empty, tf.zeros_like, warp_fn)

            return result

        for output_spec, output_image in zip(output_specs, output_images):
            if output_spec in temporal_inputs:
                input_dict[cambrian.nn.get_input_name(len(input_dict))] = _warp_temporally(output_image)

        return input_dict, output_dict
    return parse_image_ab

@ex.automain
@LogFileWriter(ex)
def main(args, _seed):
    print("python distillation.py with \"args =", args, "\"")

    tf.set_random_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    distribute_strategy = cambrian.nn.get_distribution_strategy(args["num_gpus"])
    
    run_config = tf.estimator.RunConfig(
		model_dir=args["model_dir"],
		train_distribute=distribute_strategy,
		eval_distribute=distribute_strategy,
	)

    # Get train specifiers (describes channels, paths etc.)
    a_specs, b_specs = get_specs_from_args(args, "a_input", "b_input")
    args["a_specs"], args["b_specs"] = a_specs, b_specs

    # Get eval specifiers if an eval set was given
    a_specs_eval, b_specs_eval = (None, None) if len(args["a_eval"]) == 0 or len(args["b_eval"]) == 0 else get_specs_from_args(args, "a_eval", "b_eval")
    assert a_specs_eval is None or len(a_specs) == len(a_specs_eval)
    assert b_specs_eval is None or len(b_specs) == len(b_specs_eval)

    print("A train specs:", a_specs)
    print("B train specs:", b_specs)
    print("A eval specs:", a_specs_eval)
    print("B eval specs:", b_specs_eval)
    
    model_fn = cambrian.nn.get_model_fn_ab(DistilledUNetModel, a_specs, b_specs, args=args)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=args)

    print("Start", args["mode"])

    if args["mode"] == "train":
        train_input_fn_args = cambrian.nn.InputFnArgs.train(epochs=args["epochs"], batch_size=args["batch_size"], random_flip=False)
        train_input_fn_args.augment = False
        train_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, train_input_fn_args, parse_image_fn=get_parse_image_ab_fn(a_specs, b_specs, temporal_inputs=[b_specs[i] for i in args["a_temporals"]]))
        
        # Train and eval if eval set was given, otherwise just train
        if a_specs_eval is not None and b_specs_eval is not None:
            train_spec = tf.estimator.TrainSpec(train_input_fn)

            eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
            eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs_eval, b_specs_eval, eval_input_fn_args, parse_image_fn=get_parse_image_ab_fn(a_specs_eval, b_specs_eval, temporal_inputs=[b_specs_eval[i] for i in args["a_temporals"]]))
            eval_spec = tf.estimator.EvalSpec(eval_input_fn)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            estimator.train(train_input_fn)
    elif args["mode"] == "test":
        eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
        eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, eval_input_fn_args, parse_image_fn=get_parse_image_ab_fn(a_specs, b_specs, temporal_inputs=[b_specs[i] for i in args["a_temporals"]]))
        estimator.evaluate(eval_input_fn)
    elif args["mode"] == "export":
        estimator.export_saved_model(args["export_dir"], cambrian.nn.get_serving_input_receiver_fn(a_specs + [b_specs[i] for i in args["a_temporals"]]))
    else:
        print("Unknown mode", args.mode)
