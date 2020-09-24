import tensorflow as tf


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 mask,
                                 compute_dtype='float32'):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], compute_dtype)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self,
               hparams,
               compute_dtype='float32',
               name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = hparams.num_heads
    self.d_model = hparams.d_model
    self.compute_dtype = compute_dtype

    assert self.d_model % self.num_heads == 0

    self.depth = self.d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(self.d_model)
    self.key_dense = tf.keras.layers.Dense(self.d_model)
    self.value_dense = tf.keras.layers.Dense(self.d_model)

    self.dense = tf.keras.layers.Dense(self.d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask,
                                                    self.compute_dtype)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs


def create_padding_mask(inputs):
  x, compute_dtype = inputs
  mask = tf.cast(tf.math.equal(x, 0), compute_dtype)
  return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(inputs):
  x, compute_dtype = inputs
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask([x, compute_dtype])
  return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, hparams, compute_dtype='float32'):
    super(PositionalEncoding, self).__init__()
    self.compute_dtype = compute_dtype
    self.pos_encoding = self.positional_encoding(hparams.vocab_size,
                                                 hparams.d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000,
                        (2 * (i // 2)) / tf.cast(d_model, self.compute_dtype))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=self.compute_dtype)[:, tf.newaxis],
        i=tf.range(d_model, dtype=self.compute_dtype)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=self.compute_dtype)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(hparams, compute_dtype='float32', name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, hparams.d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      hparams, compute_dtype, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(hparams.dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(
      hparams.num_units, activation=hparams.activation)(attention)
  outputs = tf.keras.layers.Dense(hparams.d_model)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(hparams, compute_dtype='float32', name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(hparams.vocab_size,
                                         hparams.d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(hparams.d_model, dtype=compute_dtype))
  embeddings = PositionalEncoding(hparams, compute_dtype)(embeddings)

  outputs = tf.keras.layers.Dropout(hparams.dropout)(embeddings)

  for i in range(hparams.num_layers):
    outputs = encoder_layer(
        hparams,
        compute_dtype,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(hparams, compute_dtype='float32', name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, hparams.d_model), name="inputs")
  enc_outputs = tf.keras.Input(
      shape=(None, hparams.d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      hparams, compute_dtype, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      hparams, compute_dtype, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(hparams.dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(
      hparams.num_units, activation=hparams.activation)(attention2)
  outputs = tf.keras.layers.Dense(hparams.d_model)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def decoder(hparams, compute_dtype='float32', name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(
      shape=(None, hparams.d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  embeddings = tf.keras.layers.Embedding(hparams.vocab_size,
                                         hparams.d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(hparams.d_model, compute_dtype))
  embeddings = PositionalEncoding(hparams, compute_dtype)(embeddings)

  outputs = tf.keras.layers.Dropout(hparams.dropout)(embeddings)

  for i in range(hparams.num_layers):
    outputs = decoder_layer(
        hparams,
        compute_dtype,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


def transformer(hparams, policy, name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')([inputs, policy.compute_dtype])
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')([dec_inputs, policy.compute_dtype])
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')([inputs, policy.compute_dtype])

  enc_outputs = encoder(hparams,
                        policy.compute_dtype)(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(hparams, policy.compute_dtype)(
      inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(
      units=hparams.vocab_size, name="outputs")(dec_outputs)

  # enable output is in float32 even training in mixed precision
  outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
