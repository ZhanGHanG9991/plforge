import sqlparse
from sqlparse.sql import TokenList
from sqlparse.tokens import Keyword
import json

def parse_skeleton(skeleton):
    """Parse the skeleton into a detailed list of tokens, handling $$ blocks."""
    # Split on the double dollar sign delimiters and filter out empty parts
    parts = [part.strip() for part in skeleton.split('$$') if part.strip()]

    tokens = []

    for part in parts:
        parsed = sqlparse.parse(part)
        if parsed:
            # Extract tokens from each parsed part
            for stmt in parsed:
                for token in stmt.tokens:
                    if not token.is_whitespace:
                        tokens.append(token)

    return tokens

class TreeNode:
    """A simple tree node for constructing an AST."""
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode(value={self.value})"

def tokens_to_ast(tokens):
    """Convert list of tokens to a simple AST."""
    root = TreeNode('ROOT')
    current_node = root
    stack = [root]

    keywords = {'BEGIN', 'DECLARE', 'LOOP', 'IF', 'THEN', 'ELSE', 'ELSIF', 'OPEN', 'CLOSE', 'FETCH', 'UPDATE'}

    for token in tokens:
        value = str(token).strip().upper()
        
        if not value:
            continue  # Skip empty tokens
        
        if token.ttype is Keyword and value.split()[0] in keywords:
            # Start a new block or statement
            node = TreeNode(value)
            current_node.add_child(node)
            
            if value in {'BEGIN', 'DECLARE', 'LOOP', 'IF'}:
                stack.append(node)
                current_node = node
            
        elif value.startswith('END'):
            # End the current block
            if stack:
                stack.pop()
            current_node = stack[-1] if stack else root
        
        else:
            # Regular statement or part of a complex statement
            node = TreeNode(value)
            current_node.add_child(node)

    return root

# Helper function to print the AST
def print_tree(node, level=0):
    if node is not None:
        print('  ' * level + str(node.value))
        for child in node.children:
            print_tree(child, level + 1)

def compute_tree_edit_distance(node1, node2):
    """Compute the edit distance between two syntax trees."""
    
    # Helper function to count nodes in a tree.
    def count_nodes(node):
        if node is None:
            return 0
        return 1 + sum(count_nodes(child) for child in node.children)
    
    if node1 is None and node2 is None:
        return 0
    
    if node1 is None:
        return count_nodes(node2)
    if node2 is None:
        return count_nodes(node1)
    
    cost = 0 if node1.value == node2.value else 1

    n1_children = len(node1.children)
    n2_children = len(node2.children)

    dp = [[0] * (n2_children + 1) for _ in range(n1_children + 1)]

    for i in range(n1_children + 1):
        for j in range(n2_children + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                dp[i][j] = min(
                    dp[i-1][j] + count_nodes(node1.children[i-1]), # delete operation
                    dp[i][j-1] + count_nodes(node2.children[j-1]), # insert operation
                    dp[i-1][j-1] + compute_tree_edit_distance(node1.children[i-1], node2.children[j-1]) # replace operation
                )

    return cost + dp[n1_children][n2_children]


def compute_edit_distance_matrix(trees1, trees2):
    """Compute the edit distance matrix for two lists of syntax trees."""
    num_trees1 = len(trees1)
    num_trees2 = len(trees2)
    distance_matrix = [[0] * num_trees2 for _ in range(num_trees1)]
    min = 99
    max = 0
    for i in range(num_trees1):
        for j in range(num_trees2):
            distance = compute_tree_edit_distance(trees1[i], trees2[j])
            distance_matrix[i][j] = distance
            if min> distance:
                min = distance
            if max < distance:
                max = distance
            
            # Check if this is the first time we encounter each distance value from 0 to 27
    if max > min:
        distance_matrix =  [[(max - value)/(max - min) for value in row] for row in distance_matrix]
    else:
        distance_matrix =  [[1 for value in row] for row in distance_matrix]
    return distance_matrix

def get_plsql_skeleton_similarity(skeleton_list1, skeleton_list2):
    tree_list1 = [tokens_to_ast(parse_skeleton(skeleton)) for skeleton in skeleton_list1]
    tree_list2 = [tokens_to_ast(parse_skeleton(skeleton)) for skeleton in skeleton_list2]
    
    distance_matrix = compute_edit_distance_matrix(tree_list1,tree_list2)
    return distance_matrix


# Main function
if __name__ == "__main__":
    # Example PL/SQL skeleton as a single-line string
    data_path = '/home/zhanghang/opt/projects/researchprojects/text2PLSQL/sftDataProcessing/template/generate_data_skeleton.json'
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    # Extract skeletons
    skeleton_list = [item['skeleton'] for item in data]
    tree_list = [tokens_to_ast(parse_skeleton(skeleton)) for skeleton in skeleton_list]
    # Compute the edit distance matrix
    distance_matrix = compute_edit_distance_matrix(tree_list,tree_list[:10])
    
    # Print the distance matrix
    print("Edit Distance Matrix:")
    with open('sim.txt', 'w', encoding='utf-8') as f:
        for row in distance_matrix:
            f.write(', '.join(map(str, row)) + '\n')
