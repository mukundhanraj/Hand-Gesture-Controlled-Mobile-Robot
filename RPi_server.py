# -*- coding: utf-8 -*-
"""
Created on Tue May 10 03:02:47 2022

@author: SG
"""

import socket

HOST = "192.168.241.5"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

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
                print('w')
            # conn.sendall(bytes('ack','utf-8'))
            