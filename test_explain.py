import psycopg2


def pg_init():
    try:
        conn = psycopg2.connect(
            database='exp', user='imdb', password='ych19960128', host='127.0.0.1', port='5433')
    except:
        print("I am unable to connect to the database")
    cursor = conn.cursor()
    return cursor


def card(query, cursor):
    cursor.execute("""EXPLAIN """ + query)
    rows = cursor.fetchall()
    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")
    return estimatedRows


if __name__ == '__main__':
    cursor = pg_init()
    query = "SELECT * FROM movie_companies, movie_info, movie_info_idx, movie_keyword, title WHERE title.id = movie_info.movie_id AND title.id = movie_keyword.movie_id AND title.id = movie_info_idx.movie_id AND title.id = movie_companies.movie_id AND movie_keyword.movie_id = movie_info.movie_id AND movie_keyword.movie_id = movie_info_idx.movie_id AND movie_keyword.movie_id = movie_companies.movie_id AND movie_info.movie_id = movie_info_idx.movie_id AND movie_info.movie_id = movie_companies.movie_id AND movie_companies.movie_id = movie_info_idx.movie_id;"
    est = card(query, cursor)
