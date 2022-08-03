import RPi.GPIO as gpio
from six.moves import input
import time
import socket

HOST = "192.168.241.9"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

def init():
    gpio.setmode(gpio.BOARD)
    gpio.setup(31,gpio.OUT)
    gpio.setup(33,gpio.OUT)
    gpio.setup(35,gpio.OUT)
    gpio.setup(37,gpio.OUT)

def gameover():
    
    gpio.output(31,False)
    gpio.output(33,False)
    gpio.output(35,False)
    gpio.output(37,False)
    
def forward(tf):
    init()
    
    gpio.output(31,True)
    gpio.output(33,False)
    
    gpio.output(35,False)
    gpio.output(37,True)
    
    time.sleep(tf)
    
    gameover()
    gpio.cleanup()
    
def backward(tf):
    init()
    
    gpio.output(31,False)
    gpio.output(33,True)
    
    gpio.output(35,True)
    gpio.output(37,False)
    
    time.sleep(tf)
    
    gameover()
    gpio.cleanup()

def left(tf):
    init()
    
    gpio.output(31,False)
    gpio.output(33,True)
    
    gpio.output(35,False)
    gpio.output(37,True)
    
    time.sleep(tf)
    
    gameover()
    gpio.cleanup()

def right(tf):
    init()
    
    gpio.output(31,True)
    gpio.output(33,False)
    
    gpio.output(35,True)
    gpio.output(37,False)
    
    time.sleep(tf)
    
    gameover()
    gpio.cleanup()
    
tf = 1


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            
            if not data:
                break
            print(data)
            if data == b'straight':
                forward(tf)
            elif data == b'left':
                left(tf)
            elif data == b'right':
                right(tf)
            elif data == b'stop':
                init()
                gameover()
                gpio.cleanup()
            else:
                print("Invalid")
                
    
    

    