import logging
from app import app
from models import init
from routes import *


@app.errorhandler(Exception)
def api_exception(e):
    logging.exception(e)
    if isinstance(e, PermissionDenied):
        return jsonify({'code': -1, 'msg': str(e)}), 403
    if isinstance(e, NeedAuth):
        return jsonify({'code': -1, 'msg': str(e)}), 401
    return jsonify({'code': -1, 'msg': str(e)}), 500


if __name__ == "__main__":
    try:
        init()
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        app.run(port=80, host="0.0.0.0")
