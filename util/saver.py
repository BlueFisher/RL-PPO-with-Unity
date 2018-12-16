import tensorflow as tf


class Saver(object):
    def __init__(self, model_path, sess):
        self.model_path = model_path
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=10)
        self.last_step = 0

    def restore_or_init(self, step=None, train_mode=True):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            if step is None:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                if train_mode:
                    self.last_step = int(ckpt.model_checkpoint_path.split('-')[1].split('.')[0])
            else:
                for c in ckpt.all_model_checkpoint_paths:
                    if f'model-{self.last_step+step}' in c:
                        self.saver.restore(self.sess, c)
                        self.last_step = step
                        break
                else:
                    paths = ', '.join(ckpt.all_model_checkpoint_paths)
                    raise Exception(f'No checkpoint step [{self.last_step+step}], available paths are [{paths}]')

    def save_graph(self, model_name=None):
        if model_name is None:
            model_name = 'raw_graph_def.pb'
        tf.train.write_graph(sess.graph_def, self.model_path, model_name, as_text=False)

    def save(self, step=None):
        assert type(step) == int
        if step is None:
            self.saver.save(self.sess, f'{self.model_path}/model.ckpt')
        else:
            self.saver.save(self.sess, f'{self.model_path}/model-{self.last_step+step}.ckpt')
