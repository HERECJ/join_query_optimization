import psycopg2


def main():
    f = open(r'/data/ygy/code_list/join2/agents/queries/crossval_sens/job_queries_simple_crossval_0_test.txt', "r")
    lines = f.readlines()
    for line in lines:
        sql_id = line.split('|')[0]
        sql = line.split('|')[1].split('$$')[0].replace("IMDB","AND")

        estimatedRows,estimatedcosts,Execution_Time=get_cost_rows(sql)
        with  open(file=r"/data/ygy/code_list/join2/agents/queries/crossval_sens/0_test_latency.txt", mode='a')as f:
            f.write(sql_id + '|' + sql + 'estimatedRows:' + estimatedRows + 'estimatedcosts:' + estimatedcosts + "Execution_Time" + Execution_Time + '\n')


def get_cost_rows(sql):
    try:
        conn = psycopg2.connect(
            database='im_database', user='imdb', password='', host='127.0.0.1', port='5432')
    except:
        print("I am unable to connect to the database")
    cursor = conn.cursor()
    cursor.execute(""" EXPLAIN ANALYZE """ + sql)
    rows = cursor.fetchall()
    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedcosts = row0[0].split("..")[1]

    print(estimatedRows)
    print(estimatedcosts)
    Execution_Time = rows[-1][0].split(":")[1].split("ms")[0].strip(' ')
    print(Execution_Time)

    return estimatedRows,estimatedcosts,Execution_Time




if __name__ == '__main__':
    main()