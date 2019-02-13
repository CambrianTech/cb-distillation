import tensorflow as tf
import cambrian

EPS = 1e-12

class DistilledUNetModel(cambrian.nn.ModelBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_general_setup = False
        self.is_train_setup = False
        self._generator_dict = None

    def _setup_general(self):
        if self.is_general_setup:
           raise Exception("General was already set up.")

        with tf.variable_scope("generator"):
            self.out_channels = self.args["out_channels"]
            self._generator_dict = create_generator(self.args, self.inputs, self.out_channels)
            self._outputs = self._generator_dict["output"]

    def _setup_train(self):
        if self.is_train_setup:
            raise Exception("Train was already set up.")

        # Split targets by channels into the seperate targets
        all_targets = []
        current_index = 0
        for spec in self.args["b_specs"]:
            all_targets.append(self.targets[:, :, :, current_index:current_index + self.out_channels])
            current_index += spec.channels
        
        print("All targets:", all_targets)

        loss = 0

        for targets in all_targets:
            if self.args["metric_loss"] == "bce":
                logits = self._generator_dict["logits"]
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
            elif self.args["metric_loss"] == "ce":
                logits = self._generator_dict["logits"]
                loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
            elif self.args["metric_loss"] == "l1":
                loss += tf.reduce_mean(tf.abs(targets - self.outputs))
            elif self.args["metric_loss"] == "mse": 
                loss += 0.5 * tf.reduce_mean(tf.square(targets - self.outputs))
            else:
                raise Exception("Unknown metric loss %s" % self.args["metric_loss"])

        with tf.name_scope("generator_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(self.args["lr_g"], self.args["beta1"], self.args["beta2"])
            gen_grads_and_vars = gen_optim.compute_gradients(loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=tf.train.get_global_step())

        # "gen_train" also includes discrim_train through control dependencies
        self._train_op = gen_train
        self._loss = loss

        # Summaries
        summaries = []

        for spec in self.args["a_specs"]:
            with tf.name_scope("inputs_summary"):
                summaries.append(tf.summary.image("inputs_%d" % spec.index, tf.image.convert_image_dtype(self.inputs[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))
        
        for spec in self.args["b_specs"]:
            with tf.name_scope("targets_summary"):
                summaries.append(tf.summary.image("targets_%d" % spec.index, tf.image.convert_image_dtype(self.targets[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))

        with tf.name_scope("outputs_summary"):
            summaries.append(tf.summary.image("output", tf.image.convert_image_dtype(self.outputs[:, :, :, :self.out_channels], dtype=tf.uint8)))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name + "/values", var))

        for grad, var in gen_grads_and_vars:
            summaries.append(tf.summary.histogram(var.op.name + "/gradients", grad))

        self._summary_op = tf.summary.merge(summaries)

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        self._setup_general()

    def set_targets(self, targets):
        super().set_targets(targets)
        self._setup_train()

def gen_conv(batch_input, out_channels, init_stddev, separable_conv):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, init_stddev)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels, init_stddev, separable_conv):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, init_stddev)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def batchnorm(inputs, init_stddev):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, init_stddev))

def layernorm(inputs):
    return tf.contrib.layers.layer_norm(inputs)

def create_generator(args, generator_inputs, generator_outputs_channels):
    output_dict = {}
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, args["ngf"], args["init_stddev"], args["separable_conv"])
        layers.append(output)

    layer_specs = [
        args["ngf"] * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        args["ngf"] * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        args["ngf"] * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        args["ngf"] * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        args["ngf"] * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        args["ngf"] * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        args["ngf"] * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = tf.nn.leaky_relu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, args["init_stddev"], args["separable_conv"])
            if not args["no_gen_bn"]:
                convolved = layernorm(convolved) if args["layer_norm"] else batchnorm(convolved, args["init_stddev"])
            layers.append(convolved)

    layer_specs = [
        (args["ngf"] * 8, 0.0),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (args["ngf"] * 8, 0.0),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (args["ngf"] * 8, 0.0),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (args["ngf"] * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (args["ngf"] * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (args["ngf"] * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (args["ngf"], 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, args["init_stddev"], args["separable_conv"])
            if not args["no_gen_bn"]:
                output = layernorm(output) if args["layer_norm"] else batchnorm(output, args["init_stddev"])

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)

        if args["angle_output"]:
            assert args["out_channels"] == 3

            # Produce 3D unit vector from 2 angles
            angles = gen_deconv(rectified, 2, args["init_stddev"], args["separable_conv"])
            angle_x = angles[:, :, :, 0:1]
            angle_y = angles[:, :, :, 1:2]

            sin_x, cos_x = tf.sin(angle_x), tf.cos(angle_x)
            sin_y, cos_y = tf.sin(angle_y), tf.cos(angle_y)
            output_x = sin_x * cos_y
            output_y = cos_x * cos_y
            output_z = sin_y

            output = tf.concat((output_x, output_y, output_z), axis=-1, )

            # [-1, 1] -> [0, 1]
            output = tf.div(output + 1., 2., name="output")
        else:
            output = gen_deconv(rectified, args["out_channels"], args["init_stddev"], args["separable_conv"])
            if args["metric_loss"] == "bce" or args["metric_loss"] == "ce":
                # Save logits when doing classification with entropy losses
                output_dict["logits"] = output
            if args["metric_loss"] == "ce":
                # Use softmax when using cross entropy
                output = tf.nn.softmax(output, name="output")
            else:
                output = tf.nn.sigmoid(output, name="output")

        print(output)
        layers.append(output)

    output_dict["output"] = layers[-1]
    
    return output_dict
