
import os
from Demo.code.test import test
import time
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    data_path = '../data'
    model_path = '../code/model.h5'

    start = time.time()
    beat_level_result, patient_level_result = test(data_path, model_path)
    end = time.time()

    print()
    print('running time: {:0.3f}'.format(end - start))

    print()
    print('beat level result')
    print('sen: {:0.3f}'.format(beat_level_result[0] / beat_level_result[1]))
    print('spe: {:0.3f}'.format(beat_level_result[2] / beat_level_result[3]))
    print('acc: {:0.3f}'.format((beat_level_result[0] + beat_level_result[2]) /
                                (beat_level_result[1] + beat_level_result[3])))

    print()
    print('patient level result')
    print('sen: {:0.3f}'.format(np.sum(patient_level_result[0]) / patient_level_result[0].shape[0]))
    print('spe: {:0.3f}'.format((patient_level_result[1].shape[0] - np.sum(patient_level_result[1])) /
                                patient_level_result[1].shape[0]))
    print('acc: {:0.3f}'.format((np.sum(patient_level_result[0]) +
                                 patient_level_result[1].shape[0] - np.sum(patient_level_result[1])) /
                                (patient_level_result[0].shape[0] + patient_level_result[1].shape[0])))

    running_time = 'running time: {:0.3f}'.format(end - start)
    beat_sen = 'sen: {:0.3f}'.format(beat_level_result[0] / beat_level_result[1])
    beat_spe = 'spe: {:0.3f}'.format(beat_level_result[2] / beat_level_result[3])
    beat_acc = 'acc: {:0.3f}'.format((beat_level_result[0] + beat_level_result[2]) /
                                (beat_level_result[1] + beat_level_result[3]))
    pa_sen = 'sen: {:0.3f}'.format(np.sum(patient_level_result[0]) / patient_level_result[0].shape[0])
    pa_spe = 'spe: {:0.3f}'.format((patient_level_result[1].shape[0] - np.sum(patient_level_result[1])) /
                                   patient_level_result[1].shape[0])
    pa_acc = 'acc: {:0.3f}'.format((np.sum(patient_level_result[0]) +
                                   patient_level_result[1].shape[0] - np.sum(patient_level_result[1])) /
                                   (patient_level_result[0].shape[0] + patient_level_result[1].shape[0]))

    save_path = '../results'
    with open(save_path + '/' + 'output.txt', mode='a+') as f:
        f.writelines(running_time + '\n')
        f.writelines('\n')

        f.writelines('beat level result\n')
        f.writelines(beat_sen + '\n')
        f.writelines(beat_spe + '\n')
        f.writelines(beat_acc + '\n')
        f.writelines('\n')

        f.writelines('patient level result\n')
        f.writelines(pa_sen + '\n')
        f.writelines(pa_spe + '\n')
        f.writelines(pa_acc + '\n')
