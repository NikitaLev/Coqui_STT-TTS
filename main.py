# This is a sample Python script.
import threading

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import STT
from fuzzywuzzy import fuzz

import TTS_Coqui
import psutil
import time

str_ex = 'the birch canoe slid on the smooth planks glue the sheet to the dark blue background it\'s easy to tell the ' \
         'depth of a well four hours of steady work faced us'


def recognize_cmd(cmd: str, name):
    rc = {'name': '', 'percent': 0}
    # print('rc', rc)
    vrt = fuzz.ratio(cmd, str_ex)
    rc['name'] = name
    rc['percent'] = vrt
    # print(x + ' x = '+str(vrt))
    return vrt


def test_cpu():
    pid = psutil.Process()
    time_end = time.time() + 20  # + 20 seconds
    cpu_count = psutil.cpu_count(logical=True)
    print('pid = ', pid)
    print('cpu_count = ', cpu_count)
    while time.time() < time_end:
        print(pid.cpu_percent(interval=1.0) / cpu_count)


def coqui_test(count, file):
    res = STT.coquiSTT(file, 'model_en.tflite')
    """    pr = recognize_cmd(res, 'Coqui')
    print('Coqui = ', res)
    print('orig   = ', str_ex)
    print('% = ', pr)
    print()"""
    TTS_Coqui.test_TTS(count, res)
    return

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    name_file1 = 'en_sample_1_16k.wav'
    params1 = {"count": 1, "filename": name_file1}
    task = threading.Thread(name="test_cpu", target=test_cpu)
    task.start()
    task_coqui_test = []
    count = 0
    while count < 5:
        count += 1
        task_coqui_test.append(threading.Thread(name="listen_coqui_" + str(count), target=coqui_test,
                                                 kwargs={"count": count, "file": name_file1}))
        task_coqui_test[-1].start()
"""    params = {"file": 'en_sample_1_16k.wav'}
    task_listen = threading.Thread(name="test_speak", target=coqui_test, kwargs=params)
    task_listen.start()"""



