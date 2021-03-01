import socket


# https://stackoverflow.com/questions/56951741/how-to-collect-video-data-from-a-dji-tello-drone-and-a-udp-server-in-python


class Drone:
    def __init__(self):
        self.command_port = 8889
        self.video_port = 11111
        self.ip = '192.168.10.1'
        self.address = (self.ip, self.command_port)
        self.response = None
        self.video_data = None

        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # initialize video streaming

        # enter command mode
        self.command_socket.sendto(b'command', self.address)

    def receive_video_data(self):
        while True:
            with self.lock:
                data, ip = self.video_socket.recvfrom(2048)
                if data:
                    print(str(data))

    def receive_response(self):
        while True:
            with self.lock:
                self.response, ip = self.socket.recvfrom(3000)
                if self.response:
                        print(str(self.response))