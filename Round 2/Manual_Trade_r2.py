import copy

EXCHANGE_MATRIX = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.3, 1, 0.46],
    [1.41, 0.61, 2.08, 1],
]
#print(EXCHANGE_MATRIX[0][1])

products = ["Pizza", "Wasabi", "Snowball", "SeaShells"]

max_amount = [0, 0, 0, 2_000_000]
initial_cap = max_amount[3]
best_route = [[], [], [], []]
result = []

# There are 5 trades
for _ in range(5):
    max_amount_new = copy.deepcopy(max_amount)
    best_route_new = copy.deepcopy(best_route)

    for target_product in range(0, 4):
        for origin_product in range(0, 4):
            quantity_target = max_amount[origin_product] * EXCHANGE_MATRIX[origin_product][target_product]
            #print(quantity_target)
            if quantity_target > max_amount_new[target_product]:
                max_amount_new[target_product] = quantity_target
                #print(max_amount_new)
                best_route_new[target_product] = best_route[origin_product] + [(origin_product, target_product)]

    max_amount = max_amount_new
    best_route = best_route_new

print("Max amounts for goods: {} is {}".format(products,max_amount),"\n")
print("Best trading strategies for goods: {} is {}".format(products, best_route),"\n")

result = [(max_amount[3], best_route[3])]
result.sort(key=lambda x: (-x[0], len(x[1])))

best_route_seashells = [(products[i], products[j]) for i, j in result[0][1]]

print("Final amount for SeaShells: {}\nAchieved with the combination of: {}".format(result[0][0], best_route_seashells),"\n")

profit_percentage = ((max_amount[3] - initial_cap)/initial_cap)*100

print("Profit percentage from the beginning 2 million SeaShells: {:.2f}% with profit: ${:.2f}"
      .format(profit_percentage, (max_amount[3] - initial_cap) ))


