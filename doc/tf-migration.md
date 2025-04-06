
-tf.set_random_seed(exper_params['seed1'])
+tf.random.set_seed(exper_params['seed1'])

tf.placeholder ❌
tf.variable_scope ❌
tf.layers.dense ❌ (moved to tf.keras.layers.Dense)
reuse=True scoping pattern ❌
tf.get_collection with graph keys ❌
tf.train.AdamOptimizer ❌ (replaced by tf.keras.optimizers.Adam)
tf.reduce_mean(tf.log(...)) pattern — numerically unstable anyway
tf.trainable_variables() tied to scopes — no longer idiomatic
