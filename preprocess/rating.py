import json

# 將MOS轉換成Level

def save_to_json(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

with open("data/label.json", 'r', encoding='utf-8') as file:
    datas = json.load(file)

min_mos = 10
max_mos = 0
for key in datas:
    data = datas[key]
    mos = float(data['mos']) 
    if mos < min_mos:
        min_mos = mos
    elif mos > max_mos:
        max_mos = mos

pivot = (max_mos - min_mos)/5
pivot = round(pivot,2)
print(max_mos, min_mos, pivot)

rating = [min_mos] * 5
for i,r in enumerate(rating):
    rating[i] = r + pivot * i

rating.append(max_mos)
rating_text = ["bad", "poor", "fair", "good", "excellent"] 

for key in datas:
    data = datas[key]
    mos = data['mos']
    for i in range(5):
        if mos >= rating[i] and mos <= rating[i+1]:
            data['rating'] = rating_text[i]
        if i==4 and data.get('rating',-1) == -1:
            print(data)
            exit(0)

save_to_json(datas, "data/label_r.json")