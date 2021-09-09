import psycopg2
from queryoptimization.QueryGraph2 import Query, Relation

"""
    cm1(): 
        calculates the cost of the given query object.
    card and card_nochache():
        gets the cardinality estimations from PostgreSQL.

̣__author__ = 'Jonas Heitz'
"""

cardinality = {}

def cm1(query,cursor,lambda_factor=1):
    r = 0.2

    # cost of table scan (1 Relation)
    if type(query) is Relation:
        costs = r * card(query,cursor)

    # cost of index join
    elif (type(query.right) is Relation) and (any({*query.right.indices} & {*query.join_condition})): # checks if query.right.indices as a mutual element with query.join_condition
        costs = cm1(query.left,cursor) + lambda_factor * card(query.left,cursor) * max(
            card(query,cursor) / card(query.left,cursor), 1)

    # cost of cross join
    elif len(query.join_condition) is 0:
        costs = cm1(query.left,cursor) * cm1(query.right,cursor)+card(query.left,cursor) * card(query.right,cursor)

    # cost of hash join
    else:
        costs = card(query,cursor) + cm1(query.left,cursor) + cm1(query.right,cursor)
    return costs

def cm2(query,cursor,lambda_factor=1):
    costs=0

    if type(query.right) is Relation and type(query.left) is Relation :
        costs = card(query,cursor)
        if costs<0:
            print("Relation Relation")
            print(query.toSql(0))

    # cost of index join
    elif (type(query.right) is Query) and (type(query.left) is Relation):
        Right=query.right
        costs = card(query, cursor)-card(Right,cursor)
        costs = card(query, cursor) - card(Right, cursor)
        if costs<0:
            print("Relation Query")
            print(card(query, cursor))
            print(card(Right, cursor))
            print(card(query, cursor)-card(Right,cursor))

            print(costs)
            print(query.toSql(0))
            print(Right.toSql(0))
            cursor.execute("EXPLAIN " + query.toSql(0))
            rows_1 = cursor.fetchall()
            print(rows_1)
            row0_1 = rows_1[0][0].split("(cost=")[1].split(' ')
            print(row0_1)
            estimatedRows = float(row0_1[0].split("..")[1])
            print(estimatedRows)
    # cost of cross join
    elif (type(query.left) is Query) and (type(query.right) is Relation):
        Left = query.left
        costs = card(query, cursor)-card(Left,cursor)
        costs = card(query, cursor) - card(Left, cursor)
        if costs<0:
            print("Query Relation")
            print(card(query, cursor))
            print(card(Left,cursor))
            print(card(query, cursor)-card(Left,cursor))

            print(costs)
            print(query.toSql(0))
            print(Left.toSql(0))
            cursor.execute("EXPLAIN " + query.toSql(0))
            rows_1 = cursor.fetchall()
            print(rows_1)
            row0_1 = rows_1[0][0].split("(cost=")[1].split(' ')
            print(row0_1)
            estimatedRows = float(row0_1[0].split("..")[1])
            print(estimatedRows)

    # cost of hash join
    elif (type(query.left) is Query) and (type(query.right) is Query):
        Left = query.left
        Right = query.right
        costs = card(query, cursor)-card(Left,cursor)-card(Right,cursor)
        if costs<0:
            print("Query Query")
            costs = card(query, cursor) -card(Left,cursor)-card(Right,cursor)
            print(card(query, cursor))
            print(card(Left, cursor))
            print(card(Right, cursor))
            print(card(query, cursor) - card(Left, cursor)-card(Right,cursor))
            print(costs)
            print(query.toSql(0))
            print(Left.toSql(0))
            print(Right.toSql(0))
            cursor.execute(" EXPLAIN " + query.toSql(0))
            print("EXPLAIN " + query.toSql(0))
            rows_1 = cursor.fetchall()
            print(rows_1)
            row0_1 = rows_1[0][0].split("(cost=")[1].split(' ')
            print(row0_1)
            estimatedRows = float(row0_1[0].split("..")[1])
            print(estimatedRows)
    else:
        costs =0



    return costs


def card_nochache(query,cursor):
    #query.toSql(0)
    cursor.execute(""" EXPLAIN """ + query.toSql(0))
    rows = cursor.fetchall()
    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")
    return float(estimatedRows)

#caches already used cardinalities
def card(query, cursor):
    global cardinality
    #print(query.toSql(0))
    if query.name in cardinality and (
            (type(query) is Relation) or (''.join(sorted(query.joined_columns)) in cardinality[query.name])):
        if type(query) is Relation:
            return int(cardinality[query.name]['card'])
        elif ''.join(sorted(query.joined_columns)) in cardinality[query.name]:
            return int(cardinality[query.name][''.join(sorted(query.joined_columns))]['card'])
    else:

        try:
            cursor.execute("""EXPLAIN """ + query.toSql(0))
            rows = cursor.fetchall()
            row0 = rows[0][0].split("(cost=")[1].split(' ')
            estimatedRows = row0[1].replace("rows=", "")
        except:
            print(query.toSql(0))

        if query.name not in cardinality:
            cardinality[query.name] = {}
        if type(query) is Relation:
            cardinality[query.name]['card'] = estimatedRows
        else:
            cardinality[query.name][''.join(sorted(query.joined_columns))] = {'card': estimatedRows}

        return float(estimatedRows)
