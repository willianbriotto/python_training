import socket
import os
import sys

HOST = '127.0.0.1'
PORT = 2000
BYTES_SIZE = 1024
REPLY = "Welcome to Client/Server Python"

sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sk.bind((HOST, PORT))
sk.listen(1)
conn,addr = sk.accept()
print "Connected by ", addr,
conn.sendall(REPLY)
while 1:
    data = conn.recv(BYTES_SIZE)
    if not data:
        continue
    print 'Server received: ', data, ' form client ', addr
conn.close()