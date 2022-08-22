import sqlite3

if __name__ == '__main__':
    print("Connecting to db and creating a table")
    con = sqlite3.connect('rtml.db', check_same_thread=False)
    cur = con.cursor()
    try:
        cur.execute('''CREATE TABLE experiments
               (experiment_id text, score_model_id text, experiment_name text, inputs text, predictions text, insert_ts text)''')
    except Exception as e:
        print(e)
    finally:
        con.close()
