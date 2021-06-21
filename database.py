import sqlite3
def createDB(DB):
    try:
        conn = sqlite3.connect(DB)
        print("Opened database successfully")
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS records (input TEXT NOT NULL, output TEXT, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        print("Tables created successfully")
        conn.close()
    except RuntimeError as e:
        print(f'Database error: {e}')

def sql_query(DB, query, param=None):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    if param == None:
        cur.execute(query)
    else:
        cur.execute(query, param)
    conn.commit()
    conn.close()


def show_data(DB, table):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(f'SELECT * from {table}')
    return cur.fetchall()