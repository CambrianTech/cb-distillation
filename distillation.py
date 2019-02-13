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

        "mode": "train",
        "model_dir": "models",
        "export_dir": "export",

        "epochs": 3000,
        
        "batch_size": 32,
        "ngf": 8,
        "ndf": 8,
        "init_stddev": 0.02,

        "crop_size": 256,
        "scale_size": 0,
        
        "lr_g": 0.0002,
        "lr_d": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,

        "metric_loss": "bce",
        
        "num_gpus": 1,

        "separable_conv": False,
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
    a_specs_eval, b_specs_eval = None, None if len(args["a_eval"]) == 0 or len(args["b_eval"]) == 0 else get_specs_from_args(args, "a_eval", "b_eval")
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
        train_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, train_input_fn_args)
        
        # Train and eval if eval set was given, otherwise just train
        if a_specs_eval is not None and b_specs_eval is not None:
            train_spec = tf.estimator.TrainSpec(train_input_fn)

            eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
            eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs_eval, b_specs_eval, eval_input_fn_args)
            eval_spec = tf.estimator.EvalSpec(eval_input_fn)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            estimator.train(train_input_fn)
    elif args["mode"] == "test":
        eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
        eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, eval_input_fn_args)
        estimator.evaluate(eval_input_fn)
    elif args["mode"] == "export":
        estimator.export_saved_model(args["export_dir"], cambrian.nn.get_serving_input_receiver_fn(a_specs))
    else:
        print("Unknown mode", args.mode)
