import io
import re
import json
import base64
from PIL import Image
from http import server
import numpy as np


class PredictionServer(server.SimpleHTTPRequestHandler):

    model = None

    def api_predict(self, body):

        res = {}
        if not self.__class__.model == None:
            img_str = re.search(r'base64,(.*)', body).group(1)
            image_bytes = io.BytesIO(base64.b64decode(img_str))
            im = Image.open(image_bytes)
            arr = np.array(im)[:,:,0:1]
            arr = (255 - arr) / 255.
            arr = arr.reshape(1, 28, 28, 1)
            predictions = self.__class__.model.predict(arr)[0]
            res['result'] = 1
            res['data'] = list(predictions.tolist())

        self.send_json(res)

    def send_json(self, body):
        body = json.dumps(body).encode('UTF8', 'replace')
        self.send_response(200, 'OK')
        self.send_header("Content-type", "application/json")
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = str(self.rfile.read(length), 'UTF8')
            if self.path.endswith('/api/predict'):
                self.api_predict(body)
            else:
                self.send_response(404, 'Not Found')
        except Exception as ex:
            self.send_error(500, str(ex))

    def do_GET(self):
        super().do_GET()
