import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import time

params = trt.TrtConversionParams(precision_mode='FP16')

converter = trt.TrtGraphConverterV2(input_saved_model_dir = 'trt_init', conversion_params=params)
converter.convert()
converter.save('trt_net')

def test():
    loaded = tf.saved_model.load('trt_net')
    infer = loaded.signatures["serving_default"]
    ls = []
    for _ in range(10):
        preds = testspeed(infer)
        ls.append(preds)
    input('press any key to continue')
    for preds in ls:
        vals = np.array(preds["dense_1"])[0][0]
        pols = np.array(preds["dense_2"])[0]
        print(pols)
        print(vals)


def testspeed(infer):
    image = tf.constant(np.random.randint(0, 2, [32, 8, 8, 3]).astype(np.float32))
    pt = time.time()
    predictions = infer(image)
    print(time.time() - pt)
    return predictions

#test()

