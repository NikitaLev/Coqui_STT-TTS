# This is a sample Python script.
import threading

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import STT
from fuzzywuzzy import fuzz

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


def coqui_test(file):
    res = STT.coquiSTT(file, 'model_en.tflite')
    pr = recognize_cmd(res, 'Coqui')
    print('Coqui = ', res)
    print('orig   = ', str_ex)
    print('% = ', pr)
    print()
    return

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    params = {"file": 'en_sample_1_16k.wav'}
    task_listen = threading.Thread(name="test_speak", target=coqui_test, kwargs=params)
    task_listen.start()



