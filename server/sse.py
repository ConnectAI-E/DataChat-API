#
#  A simple implementation of Server-Sent Events for Flask
#  that doesn't require Redis pub/sub.
#  Created On 21 November 2022
#
import logging
from time import time
from re import search
from json import dumps
from queue import Queue, Full
from flask import Response


class ServerSentEvents:
    """A simple implementation of Server-Sent Events for Flask."""

    msg_id: int = 0
    listeners = []

    def response(self):
        """Returns a response which can be passed to Flask server."""

        def stream():
            has_finished = False

            queue = Queue(1000)
            self.listeners.append(queue)

            while has_finished == False:
                msg = queue.get()

                if search("event: end", msg) or search("event: error", msg):
                    yield 'data: [DONE]\n\n'
                    has_finished = True
                else:
                    yield msg

        return Response(stream(), mimetype="text/event-stream")

    def send(self, payload = None, event: str = "data"):
        """Sends a new event to the opened channel."""

        self.msg_id = self.msg_id + 1

        if event == 'data' or event == 'role':
            if event == 'role':
                logging.info("event role %r", payload)
            msg = 'data: ' + dumps({
                'id': 'cmpl-{}'.format(self.msg_id),
                'object': 'text_completion',
                'created': int(time()),
                'choices': [{
                    'delta': {'role': payload} if event == 'role' else {'content': payload},
                    'index': 0,
                    'finish_reason': None,
                }]
            }) + '\n\n'
        else:
            msg_str = dumps(payload) if payload else "{}"
            msg = f"id: {self.msg_id}\nevent: {event}\ndata: {msg_str}\n\n"

        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
                # self.listeners[i].task_done()
            except Full as e:
                logging.error("error %r", e)
                del self.listeners[i]

        return self
