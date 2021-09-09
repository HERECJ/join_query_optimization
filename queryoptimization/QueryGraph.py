import numpy as np

from queryoptimization.subplan2tree import Tree

join_conditions = {}
join_conditions_id = {}
column_id = {}


class Query_Init():
    # global join_conditions, join_conditions_id
    # join_conditions = {}
    # join_conditions_id = {}
    # mask = []

    def __init__(self, sqlquery, schema, indices):
        global join_conditions, join_conditions_id, column_id

        schema_one_hot = np.zeros(sum(len(col)
                                      for _, col in schema.items()), dtype=int)
        relations = {}
        relations_id = {}
        pointer = 0
        # relations_cum = {}
        column_id = {}
        id_idx = 0
        for val, col, in schema.items():
            relations[val] = list(schema_one_hot.copy())
            relations_id[val] = id_idx
            id_idx += 1
            # relations_cum[val] = pointer
            for i in range(0, len(col)):
                column_name = val + '.' + col[i]
                column_id[column_name] = pointer
                relations[val][pointer] = 1
                pointer += 1
        self.actions = self.get_Actions(
            sqlquery, relations, relations_id, schema, indices)
        join_conditions, join_conditions_id, self.link_mtx = self.get_Conditions(
            sqlquery, self.actions, relations_id, column_id)
        # self.actions, join_conditions, join_conditions_id, self.link_mtx = self.get_Actions_Conditions(sqlquery, relations, relations_id, schema, indices, column_id)

        for i in range(len(self.actions), len(schema)):
            self.actions.append(EmptyQuery(schema_one_hot))

        # sorted action space
        sorted_actions = []
        for key, value in schema.items():
            flag = True
            for action in self.actions:
                name = action.name
                if name == key:
                    sorted_actions.append(action)
                    flag = False
                    break

            if flag: sorted_actions.append(EmptyQuery(schema_one_hot))
        self.actions = sorted_actions

    # def get_Actions_Conditions(self, sql, masks, relations_id, schema, indices, column_id):
    #     action_space = []
    #     try:
    #         relations = sql.split('FROM')[1].split(
    #             'WHERE')[0].replace(" ", "").split(',')
    #     except Exception:
    #         relations = sql.split('FROM')[1].replace(" ", "")
    #         pass

    #     for r in relations:
    #         r = r.replace("AS", " AS ")
    #         r_split = r.split(" AS ")
    #         # orig_table_name = r_split[0]
    #         join_table_name = r_split[-1]  # if no AS, then the final
    #         action_space.append(Relation(
    #             r, join_table_name, masks[join_table_name], schema[join_table_name], relations_id[join_table_name], indices))

    #     num_relations = len(relations_id)
    #     link_mtx = np.zeros((num_relations, num_relations), dtype=float)
    #     try:
    #         all_conditions = sql.split('WHERE')[1].split('AND')
    #     except Exception:
    #         all_conditions = []
    #         pass

    #     for condition in all_conditions:
    #         if condition.startswith(' |'):
    #             # |0.8|table.col IN LIKE ...
    #             selectivity = float(condition.split('|')[1])
    #             clause = ('|').join(condition.split('|')[2:])
    #             table_name = clause.split()[0].split('.')[0]
    #             for action in action_space:
    #                 if action.name == table_name:
    #                     action.set_clause(clause)
    #         elif "=" in condition:
    #             clauses = condition.split('=')
    #             element = []
    #             for clause in clauses:
    #                 element.append(clause.replace(
    #                     "\n", "").replace(" ", "").split("."))
    #             try:
    #                 l_table = element[0][0]
    #                 r_table = element[1][0]
    #                 cond_tables = '&'.join(
    #                     sorted([element[0][0]] + [element[1][0]]))
    #                 l_col = element[0][0] + '.' + element[0][1]
    #                 r_col = element[1][0] + '.' + element[1][1]
    #                 try:
    #                     l_rel_id = relations_id[l_table]
    #                     r_rel_id = relations_id[r_table]
    #                     l_id = column_id[l_col]
    #                     r_id = column_id[r_col]
    #                 except Exception:
    #                     pass
    #                 join_conditions[cond_tables] = [l_col, r_col]
    #                 join_conditions_id[cond_tables] = [l_id, r_id]
    #                 link_mtx[l_rel_id][r_rel_id] = 1.0
    #                 link_mtx[r_rel_id][l_rel_id] = 1.0

    #             except Exception:
    #                 pass
    #     return action_space, join_conditions, join_conditions_id, link_mtx

    def get_Actions(self, sql, masks, relations_id, schema, indices):
        action_space = []
        try:
            relations = sql.split('FROM')[1].split(
                'WHERE')[0].replace(" ", "").split(',')
        except Exception:
            relations = sql.split('FROM')[1].replace(" ", "")
            pass

        for r in relations:
            r = r.replace("AS", " AS ")
            r_split = r.split(" AS ")
            # orig_table_name = r_split[0]
            join_table_name = r_split[-1]  # if no AS, then the final
            action_space.append(Relation(
                r, join_table_name, masks[join_table_name], schema[join_table_name], relations_id[join_table_name],
                indices))

        return action_space

    def get_Conditions(self, sql, action_space, relations_id, column_id):
        num_relations = len(relations_id)
        link_mtx = np.zeros((num_relations, num_relations), dtype=float)
        try:
            all_conditions = sql.split('WHERE')[1].split('AND')
        except Exception:
            all_conditions = []
            pass

        for condition in all_conditions:
            if condition.startswith(' |'):
                # |0.8|table.col IN LIKE ...
                selectivity = float(condition.split('|')[1])
                clause = ('|').join(condition.split('|')[2:])
                table_name = clause.split()[0].split('.')[0]
                for action in action_space:
                    if action.name == table_name:
                        action.set_clause(clause)
            elif "=" in condition:
                clauses = condition.split('=')
                element = []
                for clause in clauses:
                    element.append(clause.replace(
                        "\n", "").replace(" ", "").split("."))
                try:
                    l_table = element[0][0]
                    r_table = element[1][0]
                    l_col = element[0][0] + '.' + element[0][1]
                    r_col = element[1][0] + '.' + element[1][1]
                    if r_table < l_table:
                        l_table, r_table = r_table, l_table
                        l_col, r_col = r_col, l_col

                    cond_tables = '&'.join(sorted([l_table] + [r_table]))
                    # cond_tables = '&'.join(sorted([element[0][0]] + [element[1][0]]))
                    try:
                        l_rel_id = relations_id[l_table]
                        r_rel_id = relations_id[r_table]
                        l_id = column_id[l_col]
                        r_id = column_id[r_col]
                    except Exception:
                        pass
                    join_conditions[cond_tables] = [l_col, r_col]
                    join_conditions_id[cond_tables] = [l_id, r_id]
                    link_mtx[l_rel_id][r_rel_id] = 1.0
                    link_mtx[r_rel_id][l_rel_id] = 1.0

                except Exception:
                    pass
        return join_conditions, join_conditions_id, link_mtx


class Relation(object):
    id = None
    name = ''
    sql_name = ''
    mask = []
    columns = []
    clauses = []

    def __init__(self, name, tab_name, mask, columns, table_id, indices):
        self.name = tab_name
        self.sql_name = name
        self.mask = mask
        self.id = table_id
        self.indices = []
        self.columns = []
        self.clauses = []

        for column in columns:
            column_name = tab_name + '.' + column
            self.columns.append(column_name)

        # 这一块代码 计算cost时候会使用
        for i in indices:
            if " AS " in self.name:
                table = self.name.split(" AS ")[1]
            else:
                table = self.name
            if i.split('.')[0] == table:
                self.indices.append(i)

    def set_clause(self, selection):
        """
            Save the selection for generate the sql
        """
        table_name = selection.split(" ")[0].split('.')[0]
        assert table_name == self.name, ValueError(
            'Does not match the table name!!!')
        self.clauses.append(selection)

    def clause_to_sql(self):
        if len(self.clauses) == 0:
            return self.sql_name
        else:
            sql = '( SELECT * FROM ' + self.sql_name + ' WHERE '
            clauses = ' AND '.join(self.clauses)
            sql += clauses
            sql += ') AS ' + self.name
            self.clauses = []
            return sql

    def toSql(self, level):
        return "SELECT * FROM " + self.sql_name + ";"


class EmptyQuery(object):
    name = 'EmptyQuery'
    mask = []

    def __init__(self, mask):
        self.mask = mask


class Query(object):
    left = None
    right = None
    name = ''
    join_condition = {}
    join_condition_id = {}
    joined_columns = []
    mask = []
    columns = []
    aliasflag = True

    # aliasflag = False

    def __init__(self, left, right):
        global join_conditions, join_conditions_id
        self.joined_columns = []

        self.left = left
        self.right = right

        lname = self.left.name
        lname_list = lname.split("&")

        rname = self.right.name
        rname_list = rname.split("&")

        self.name = '&'.join(sorted(lname_list + rname_list))
        self.mask = [x | y for (x, y) in zip(left.mask, right.mask)]

        if self.name in join_conditions:
            self.join_condition = join_conditions[self.name]
            self.join_condition_id = join_conditions_id[self.name]
            for i in self.join_condition:
                self.joined_columns.append(i.split('.')[1])
        if type(self.left) is Query:
            self.joined_columns = self.joined_columns + self.left.joined_columns

        if type(self.right) is Query:
            self.joined_columns = self.joined_columns + self.right.joined_columns

        self.columns = []
        tmpcolumns = []
        for c in left.columns:
            self.columns.append(lname + '.' + c.split('.')[1])
            tmpcolumns.append(c.split('.')[1])

        for c in right.columns:
            if " AS " in c:
                c = rname + "." + c.split(" AS ")[1]
            if c.split('.')[1] in tmpcolumns:
                new_column = rname + "." + str(c.split('.')[0].split("&")[0] + c.split('.')[0].split("&")[-1]) + "$" + \
                             c.split('.')[1]
                while new_column.split('.')[1] in tmpcolumns:
                    new_column = new_column + "_tmp"
                    import pdb;
                    pdb.set_trace()
                tmpcolumns.append(new_column.split('.')[1])
                self.columns.append(rname + "." + c.split('.')[1] + " AS " + new_column.split('.')[1])
                for key, val in join_conditions.items():
                    newval = []
                    for v in val:
                        if v == rname + "." + c.split('.')[1]:
                            newval.append(v.replace(rname + "." + c.split('.')[1], new_column))
                            if "kind_type" in new_column: print(new_column)
                        else:
                            newval.append(v)
                    join_conditions[key] = newval
            else:
                self.columns.append(rname + "." + c.split(".")[1])
                tmpcolumns.append(c.split('.')[1])

        join_conditions, join_conditions_id = self.deleteJoinCondition(lname, rname, join_conditions,
                                                                       join_conditions_id)

        join_conditions, join_conditions_id = self.changeJoinConditions(lname, rname, self.name, join_conditions,
                                                                        join_conditions_id)

    def changeJoinConditions(self, relA, relB, relnew, join_conditions, join_conditions_id):
        conditions = {}
        conditions_id = {}
        relB = relB.split('&')
        relA = relA.split('&')
        for key, value in join_conditions.items():
            if set(relB).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&') + relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relB).issubset(v.split('.')[0].split('&')):
                        value2.append(
                            '&'.join(np.unique(sorted(v.split('.')[0].split('&') + relnew.split('&')))) + '.' +
                            v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]
            elif set(relA).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&') + relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relA).issubset(v.split('.')[0].split('&')):
                        value2.append(
                            '&'.join(np.unique(sorted(v.split('.')[0].split('&') + relnew.split('&')))) + '.' +
                            v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]

            else:
                if key in conditions:
                    conditions[key] = conditions[key] + value
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[key] = value
                    conditions_id[key] = join_conditions_id[key]

        return conditions, conditions_id

    def deleteJoinCondition(self, relA, relB, jc, jc_id):
        conditions = dict(jc)
        conditions_id = dict(jc_id)
        try:
            join_key = '&'.join(np.unique(sorted(relA.split('&') + relB.split('&'))))
            # del conditions[join_key]
            conditions.pop(join_key)
            # del conditions_id[join_key]
            conditions_id.pop(join_key)
        except Exception:
            pass
        return conditions, conditions_id

    def _toSql(self, level):
        if self.aliasflag:
            sql = 'SELECT '
            i = False
            for c in self.columns:
                if i:
                    sql += ", "
                i = True
                sql += c
            sql += ' FROM '

        else:
            sql = 'SELECT * FROM '

        if type(self.left) is Relation and type(self.right) is Relation:
            subsql_left = self.left.clause_to_sql()
            subsql_right = self.right.clause_to_sql()

        elif type(self.left) is Relation and type(self.right) is Query:
            subsql_left = self.left.clause_to_sql()
            subsql_right = self.right._toSql(1)

        elif type(self.left) is Query and type(self.right) is Relation:
            subsql_left = self.left._toSql(1)
            subsql_right = self.right.clause_to_sql()

        elif type(self.left) is Query and type(self.right) is Query:
            subsql_left = self.left._toSql(1)
            subsql_right = self.right._toSql(1)
        else:
            return ""

        if len(self.join_condition) is not 0:
            sql_join_condition = self.join_condition[0] + '=' + self.join_condition[1]
            if len(self.join_condition) > 2:
                for i in range(2, len(self.join_condition), 2):
                    sql_join_condition = sql_join_condition + ' AND ' + self.join_condition[i] + '=' + \
                                         self.join_condition[i + 1]
            sql += subsql_left + ' INNER JOIN ' + subsql_right + ' ON (' + sql_join_condition + ')'
        else:
            sql += subsql_left + ' CROSS JOIN ' + subsql_right

        if level is 1: sql = '(' + sql + ') AS ' + self.name + ' '

        return sql

    def toSql(self, level):
        sql = self._toSql(0)
        return sql.replace('&', '').replace('$', '')

    def to_tree_structure(self):
        tree = Tree()
        # tree.index = index

        if type(self.left) is Relation and type(self.right) is Relation:
            tree.l_name = self.left.name
            tree.r_name = self.right.name
            tree.l_table_id = self.left.id
            tree.r_table_id = self.right.id

        elif type(self.left) is Relation and type(self.right) is Query:
            tree.l_name = self.left.name
            tree.l_table_id = self.left.id
            tree.r_name = self.right.to_tree_structure()
            # tree.r_name = self.right.to_tree_structure(2 * index + 2)

        elif type(self.left) is Query and type(self.right) is Relation:
            tree.l_name = self.left.to_tree_structure()
            # tree.l_name = self.left.to_tree_structure(2 * index + 1)
            tree.r_name = self.right.name
            tree.r_table_id = self.right.id

        elif type(self.left) is Query and type(self.right) is Query:
            tree.l_name = self.left.to_tree_structure()
            tree.r_name = self.right.to_tree_structure()
            # tree.l_name = self.left.to_tree_structure(2 * index + 1)
            # tree.r_name = self.right.to_tree_structure(2 * index + 2)

        else:
            raise ValueError("Not supported (Sub)Query!")

        # self.join_condition_id
        if len(self.join_condition) is not 0:
            tree.l_column.append(self.join_condition[0])
            tree.r_column.append(self.join_condition[1])
            tree.l_column_id.append(self.join_condition_id[0])
            tree.r_column_id.append(self.join_condition_id[1])

            if len(self.join_condition) > 2:
                for i in range(2, len(self.join_condition), 2):
                    tree.l_column.append(self.join_condition[i])
                    tree.r_column.append(self.join_condition[i + 1])
                    tree.l_column_id.append(self.join_condition_id[i])
                    tree.r_column_id.append(self.join_condition_id[i + 1])
        else:
            raise ValueError("No Candidate Join Conditions!")

        return tree


class Rel_Columns(object):
    name = ''
    id = None
    selectivity = None

    def __init__(self, name, id, selectivity):
        self.name = name
        self.id = id
        self.selectivity = selectivity
        self.table = name.spilt('.')[0]


def getJoinConditions():
    return join_conditions


if __name__ == '__main__':
    import random

    schema = {
        "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                     "md5sum"],
        "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code",
                      "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
        "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"],
        "comp_cast_type": ["id", "kind"],
        "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
        "company_type": ["id", "kind"],
        "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
        "info_type": ["id", "info"],
        "keyword": ["id", "keyword", "phonetic_code"],
        "kind_type": ["id", "kind"],
        "link_type": ["id", "link"],
        "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
        "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
        "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                 "md5sum"],
        "person_info": ["id", "person_id", "info_type_id", "info", "note"],
        "role_type": ["id", "role"],
        "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                  "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
        # "comp_cast_type2": [],
        # "company_name2": [],
        # "info_type2": [],
        # "kind_type2": [],
        # "movie_companies2": [],
        # "movie_info_idx2": [],
        # "title2": []
        "comp_cast_type2": ["id", "kind"],
        "company_name2": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
        "info_type2": ["id", "info"],
        "kind_type2": ["id", "kind"],
        "movie_companies2": ["id", "movie_id", "company_id", "company_type_id", "note"],
        "movie_info_idx2": ["id", "movie_id", "info_type_id", "info", "note"],
        "title2": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                   "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"]
    }
    primary = ['akaname.md5sum', 'akaname2.md5sum', 'akaname.name',
               'akaname2.  name', 'akaname.surname_pcode', 'akaname2.surname_pcode', 'akaname.name_pcode_cf',
               'akaname2.name_pcode_cf', 'akaname.name_pcode_nf', 'akaname2.name_pcode_nf', 'akaname.person_id',
               'akaname2.person_id', 'akaname.id', 'akaname2.id', 'akatitle.episode_of_id', 'akatitle2.episode_of_id',
               'akatitle.kind_id', 'akatitle2.kind_id', 'akatitle.md5sum', 'akatitle2.md5sum', 'akatitle.movie_id',
               'akatitle2.movie_id', 'akatitle.phonetic_code', 'akatitle2.phonetic_code', 'akatitle.title',
               'akatitle2.title', 'akatitle.production_year', 'akatitle2.production_year', 'akatitle.id',
               'akatitle2.id', 'castinfo.person_role_id', 'castinfo2.person_role_id', 'castinfo.movie_id',
               'castinfo2.movie_id', 'castinfo.person_id', 'castinfo2.person_id', 'castinfo.role_id',
               'castinfo2.role_id', 'castinfo.id', 'castinfo2.id', 'charname.imdb_id', 'charname2.imdb_id',
               'charname.md5sum', 'charname2.md5sum', 'charname.name', 'charname2.name', 'charname.surname_pcode',
               'charname2.surname_pcode', 'charname.name_pcode_nf', 'charname2.name_pcode_nf', 'charname.id',
               'charname2.id', 'compcasttype.kind', 'compcasttype2.kind', 'compcasttype.id', 'compcasttype2.id',
               'companyname.country_code', 'companyname2.country_code', 'companyname.imdb_id', 'companyname2.imdb_id',
               'companyname.md5sum', 'companyname2.md5sum', 'companyname.name', 'companyname2.name',
               'companyname.name_pcode_nf', 'companyname2.name_pcode_nf', 'companyname.name_pcode_sf',
               'companyname2.name_pcode_sf', 'companyname.id', 'companyname2.id', 'companytype.kind',
               'companytype2.kind', 'companytype.id', 'companytype2.id', 'completecast.movie_id',
               'completecast2.movie_id', 'completecast.subject_id', 'completecast2.subject_id', 'completecast.id',
               'completecast2.id', 'infotype.info', 'infotype2.info', 'infotype.id', 'infotype2.id', 'keyword.keyword',
               'keyword2.keyword', 'keyword.phonetic_code', 'keyword2.phonetic_code', 'keyword.id', 'keyword2.id',
               'kindtype.kind', 'kindtype2.kind', 'kindtype.id', 'kindtype2.id', 'linktype.link', 'linktype2.link',
               'linktype.id', 'linktype2.id', 'moviecompanies.company_id', 'moviecompanies2.company_id',
               'moviecompanies.company_type_id', 'moviecompanies2.company_type_id', 'moviecompanies.movie_id',
               'moviecompanies2.movie_id', 'moviecompanies.id', 'moviecompanies2.id', 'movieinfo.info_type_id',
               'movieinfo2.info_type_id', 'movieinfo.movie_id', 'movieinfo2.movie_id', 'movieinfo.id', 'movieinfo2.id',
               'movieinfoidx.id', 'movieinfoidx2.id', 'moviekeyword.keyword_id', 'moviekeyword2.keyword_id',
               'moviekeyword.movie_id', 'moviekeyword2.movie_id', 'moviekeyword.id', 'moviekeyword2.id',
               'movielink.linked_movie_id', 'movielink2.linked_movie_id', 'movielink.link_type_id',
               'movielink2.link_type_id', 'movielink.movie_id', 'movielink2.movie_id', 'movielink.id', 'movielink2.id',
               'name.gender', 'name2.gender', 'name.imdb_id', 'name2.imdb_id', 'name.md5sum', 'name2.md5sum',
               'name.name', 'name2.name', 'name.surname_pcode', 'name2.surname_pcode', 'name.name_pcode_cf',
               'name2.name_pcode_cf', 'name.name_pcode_nf', 'name2.name_pcode_nf', 'name.id', 'name2.id',
               'personinfo.info_type_id', 'personinfo2.info_type_id', 'personinfo.person_id', 'personinfo2.person_id',
               'personinfo.id', 'personinfo2.id', 'roletype.id', 'roletype2.id', 'roletype.role', 'roletype2.role',
               'title.episode_nr', 'title2.episode_nr', 'title.episode_of_id', 'title2.episode_of_id', 'title.imdb_id',
               'title2.imdb_id', 'title.kind_id', 'title2.kind_id', 'title.md5sum', 'title2.md5sum',
               'title.phonetic_code', 'title2.phonetic_code', 'title.season_nr', 'title2.season_nr', 'title.title',
               'title2.title', 'title.production_year', 'title2.production_year', 'title.id', 'title2.id']

    sql = random.choice(list(open(
        "/data0/chenjin/join_query_optimization/agents/queries/crossval_sens/job_queries_simple_crossval_0_train.txt")))
    query = Query_Init(sql, schema, primary)