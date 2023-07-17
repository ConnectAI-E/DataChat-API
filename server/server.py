import logging
from app import app
from models import db, text, Embedding
from routes import *


@app.errorhandler(Exception)
def api_exception(e):
    import logging
    logging.exception(e)
    return jsonify({'code': -1, 'msg': str(e)}), 500


if __name__ == "__main__":
    from pgvector.psycopg2 import register_vector
    with app.app_context():
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        try:
            pass
            # index = db.Index('my_index', Embedding.embedding,
            #     postgresql_using='ivfflat',
            #     # postgresql_with={'lists': 100},
            #     postgresql_ops={'embedding': 'vector_l2_ops'}
            # )
            # index.create(db.engine)
        except Exception as e:
            logging.error(e)
        register_vector(db.engine.raw_connection())
        db.create_all()

