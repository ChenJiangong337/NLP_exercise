# Filename：test.py

from wxpy import *


@bot.register()
def print_messages(msg):
    print(msg)


