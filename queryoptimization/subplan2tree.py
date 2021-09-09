# This file (1)define the tree structure (2)get the tree structure from the subquery(subplan) for step(state,obj)

class Tree():
    def __init__(self):
        # self.parent = None
        self.left = None
        self.index = None
        self.l_table = None
        self.l_table_id = None
        self.l_name = None
        self.l_column = []
        self.l_column_id = []
        self.l_column_embed = []
        self.l_table_embed = None

        self.right = None
        self.r_table = None
        self.r_table_id = None
        self.r_name = None
        self.r_column = []
        self.r_column_id = []
        self.r_column_embed = []
        self.r_table_embed = None
    
    # def is_root(self, node):
    #     return True if node.parent is None else False
    
# def construct_tree()

# if __name__ == '__main__':
    
    

