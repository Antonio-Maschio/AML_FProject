def is_valid(x, y, grid, visited):
    rows = len(grid)
    cols = len(grid[0])
    return 0 <= x < rows and 0 <= y < cols and grid[x][y] == 1 and not visited[x][y]

def dfs_count_paths(grid, start, end, visited, path_length, path_lengths):
    x, y = start
    if start == end:
        path_lengths.append(path_length)
        return 1
    visited[x][y] = True
    path_count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny, grid, visited):
            path_count += dfs_count_paths(grid, (nx, ny), end, visited, path_length + 1, path_lengths)
    
    visited[x][y] = False
    return path_count

def count_paths_and_lengths(grid, start, end):
    if not grid or not start or not end:
        return 0, []
    if grid[start[0]][start[1]] != 1 or grid[end[0]][end[1]] != 1:
        return 0, []
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    path_lengths = []
    total_paths = dfs_count_paths(grid, start, end, visited, 0, path_lengths)
    return total_paths, path_lengths

# Example usage:
grid = [
[1, 0, 1, 1, 1],
[1, 0, 1, 0, 1],
[1, 1, 1, 0, 1],
[0, 0, 1, 0, 1],
[1, 1, 1, 1, 1]
]


start = (len(grid) - 1, len(grid[0]) // 2)
end = (0, len(grid[0]) // 2)

# Adjust start position if it initially points to a cell with 0
if grid[start[0]][start[1]] == 0:
    guard = True
    while guard:
        start = (start[0] - 1, start[1])
        if start[0] < 0 or grid[start[0]][start[1]] == 1:
            guard = False
    if start[0] < 0:
        raise ValueError("No valid start position found")


total_paths, path_lengths = count_paths_and_lengths(grid, start, end)

print(f"Total number of paths: {total_paths}")
print(f"Lengths of paths: {path_lengths}")
