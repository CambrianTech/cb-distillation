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

        gen_loss_metric = 0

        # Metric loss
        for targets in all_targets:
            if self.args["metric_loss"] == "bce":
                logits = self._generator_dict["logits"]
                gen_loss_metric += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
            elif self.args["metric_loss"] == "ce":
                logits = self._generator_dict["logits"]
                gen_loss_metric += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=logits))
            elif self.args["metric_loss"] == "l1":
                gen_loss_metric += tf.reduce_mean(tf.abs(targets - self.outputs))
            elif self.args["metric_loss"] == "mse": 
                gen_loss_metric += 0.5 * tf.reduce_mean(tf.square(targets - self.outputs))
            else:
                raise Exception("Unknown metric loss %s" % self.args["metric_loss"])

        # GAN loss only over real targets
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                predict_real = create_discriminator(self.args, self.inputs, all_targets[0])

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                predict_fake = create_discriminator(self.args, self.inputs, self.outputs)

        disc_loss = tf.reduce_mean(-(tf.log(tf.sigmoid(predict_real) + EPS) + tf.log(1 - tf.sigmoid(predict_fake) + EPS)))
        gen_loss_gan = tf.reduce_mean(-tf.log(tf.sigmoid(predict_fake) + EPS))

        loss = self.args["metric_weight"] * gen_loss_metric + gen_loss_gan

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.args["lr_d"], self.args["beta1"], self.args["beta2"])
            discrim_grads_and_vars = discrim_optim.compute_gradients(disc_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
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

        index_offset = sum(map(lambda spec: spec.channels, self.args["a_specs"]))
        for spec in [self.args["b_specs"][i] for i in self.args["a_temporals"]]:
            with tf.name_scope("inputs_warped_summary"):
                summaries.append(tf.summary.image("inputs_%d_warped" % spec.index, tf.image.convert_image_dtype(self.inputs[:, :, :, index_offset:index_offset+spec.channels], dtype=tf.uint8)))
                index_offset += spec.channels
        
        for spec in self.args["b_specs"]:
            with tf.name_scope("targets_summary"):
                summaries.append(tf.summary.image("targets_%d" % spec.index, tf.image.convert_image_dtype(self.targets[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))

        with tf.name_scope("outputs_summary"):
            summaries.append(tf.summary.image("output", tf.image.convert_image_dtype(self.outputs[:, :, :, :self.out_channels], dtype=tf.uint8)))

        with tf.name_scope("scalar_summaries"):
            summaries.append(tf.summary.scalar("discriminator_loss", disc_loss))
            summaries.append(tf.summary.scalar("generator_loss_gan", gen_loss_gan))
            summaries.append(tf.summary.scalar("generator_loss_metric", gen_loss_metric))

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

def discrim_conv(batch_input, out_channels, stride):
    initializer = tf.orthogonal_initializer()
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=initializer)

def gen_down_conv(batch_input, out_channels, kernel_size, strides):
    initializer = tf.orthogonal_initializer()
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=kernel_size, strides=(strides, strides), padding="same", kernel_initializer=initializer, activation=tf.nn.elu)

def gen_up_conv(batch_input, out_channels, kernel_size, strides, activation=tf.nn.elu):
    initializer = tf.orthogonal_initializer()
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=(strides, strides), padding="same", kernel_initializer=initializer, activation=activation)

def gen_res_conv(batch_input, squeeze_div):
    initializer = tf.orthogonal_initializer()
    out_channels = batch_input.shape.as_list()[-1]
    res_channels = out_channels / squeeze_div
    y = tf.layers.conv2d(batch_input, res_channels, kernel_size=1, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, res_channels, kernel_size=3, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, out_channels, kernel_size=1, padding="SAME", kernel_initializer=initializer)
    return tf.nn.elu(y + batch_input)

def gen_ref_conv(color_input, aux_output, filters):
    initializer = tf.orthogonal_initializer()
    out_channels = aux_output.shape.as_list()[-1]

    combined = tf.concat([color_input, aux_output], axis=-1)

    x1 = tf.layers.conv2d(combined, filters, kernel_size=1, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)

    y = tf.layers.conv2d(x1, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)

    x2 = tf.layers.conv2d(tf.concat([x1, y], axis=-1), filters, kernel_size=1, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    x2 = batchnorm(x2)

    y = tf.layers.conv2d(x2, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)
    y = tf.layers.conv2d(y, filters, kernel_size=4, padding="SAME", kernel_initializer=initializer, activation=tf.nn.elu)

    return tf.layers.conv2d(tf.concat([x1, x2, y], axis=-1), out_channels, kernel_size=1, padding="SAME", kernel_initializer=initializer)

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.ones_initializer())

def layernorm(inputs):
    return tf.contrib.layers.layer_norm(inputs)

def create_generator(args, generator_inputs, generator_outputs_channels):
    output_dict = {}
    ngf = args["ngf"]

    with tf.variable_scope("down_convs"):
        down_1 = gen_down_conv(generator_inputs, out_channels=ngf, kernel_size=8, strides=4) # 64, c32
        down_1 = batchnorm(down_1)
        down_2 = gen_down_conv(down_1, out_channels=2*ngf, kernel_size=4, strides=2) # 32, c64
        down_2 = batchnorm(down_2)
        down_3 = gen_down_conv(down_2, out_channels=4*ngf, kernel_size=4, strides=2) # 16, c128
        down_3 = batchnorm(down_3)

    with tf.variable_scope("res_convs"):
        res_output = down_3
        for _ in range(21):
            res_output = gen_res_conv(res_output, squeeze_div=16)
            res_output = batchnorm(res_output)

    with tf.variable_scope("up_convs"):
        up_3 = gen_up_conv(tf.concat([down_3, res_output], axis=-1), out_channels=2*ngf, kernel_size=4, strides=2)
        up_3 = batchnorm(up_3)
        up_2 = gen_up_conv(tf.concat([down_2, up_3], axis=-1), out_channels=ngf, kernel_size=4, strides=2)
        up_2 = batchnorm(up_2)
        up_1 = gen_up_conv(tf.concat([down_1, up_2], axis=-1), out_channels=generator_outputs_channels, kernel_size=8, strides=4, activation=tf.nn.sigmoid)

    with tf.variable_scope("refinement"):
        refined = gen_ref_conv(generator_inputs, up_1, filters=ngf)

    with tf.variable_scope("output"):
        # Save logits when doing classification with entropy losses
        if args["metric_loss"] == "bce" or args["metric_loss"] == "ce":
            output_dict["logits"] = refined

        # Use softmax when using cross entropy, sigmoid otherwise
        if args["metric_loss"] == "ce":
            output = tf.nn.softmax(refined, name="output")
        else:
            output = tf.nn.sigmoid(refined, name="output")

    output_dict["output"] = output
    
    return output_dict

def create_discriminator(args, discrim_inputs, discrim_targets):
    n_layers = 4
    layers = []

    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(inputs, args["ndf"], 2)
        rectified = tf.nn.elu(convolved)
        layers.append(rectified)

    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = args["ndf"] * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride)
            if not args["no_disc_bn"]:
                convolved = layernorm(convolved) if args["layer_norm"] else batchnorm(convolved)
            rectified = tf.nn.elu(convolved)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = discrim_conv(rectified, 1, 1)
        layers.append(output)

    return layers[-1]