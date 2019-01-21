from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(
    input_graph='tmp/raw_graph_def.pb',
    input_binary=True,
    input_checkpoint='tmp/model.ckpt',
    output_node_names='actor/action,actor/vector_observations',
    output_graph='tmp/model.bytes',
    clear_devices=True, initializer_nodes='', input_saver='',
    restore_op_name='save/restore_all',
    filename_tensor_name='save/Const:0')
