import json
import os
import time
import cherrypy
from cherrypy import log
import yaml

import psutil

from models.text.extract_description import process_text
from models.generate_image import generate_images

process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open("config.yml"))




def process_api_request(body):
    """
    This method is for extracting json parameters and processing the actual api request.
    All api format and logic is to be kept in here.
    :param body:
    :return:
    """
    # print("CUDA:", os.environ.get('CUDA_PATH'))
    # print("CUDA:", os.environ.get('CUDA_HOME'))
    # print("CUDA_HOME:", CUDA_HOME)
    #
    # version_file_path = os.path.join(CUDA_HOME, "version.txt")
    # try:
    #     with open(version_file_path) as f:
    #         print("cuda version:", f.readline())
    # except:
    #     pass

    text = body.get('text')

    start = time.time()

    descriptions = process_text(text)
    results = generate_images(descriptions)
    # search(descriptions)

    time_spent = time.time() - start
    print("Completed api call.Time spent {0:.3f} s".format(time_spent))

    # print(results[0])
    # result = {
    #     'result': "OK"
    # }
    return json.dumps({
        'result': results
    })


class ApiServerController(object):
    @cherrypy.expose('/health')
    def health(self):
        result = {
            "status": "OK",  # TODO when is status not ok?
            "info": {
                "mem": "{0:.3f} MiB".format(process.memory_info().rss / 1024.0 / 1024.0),
                "cpu": process.cpu_percent(),
                "threads": len(process.threads())
            }
        }
        return json.dumps(result).encode("utf-8")

    @cherrypy.expose('/t2i-person')
    def t2i_person(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_api_request(body).encode("utf-8")


if __name__ == '__main__':
    cherrypy.tree.mount(ApiServerController(), '/')

    cherrypy.config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [('Content-Type', 'application/json;encoding=utf-8')],
    })

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
