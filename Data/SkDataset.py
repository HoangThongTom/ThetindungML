from sklearn.datasets import fetch_openml

credit_data = fetch_openml('credit-g', version=1, as_frame=True)# trả về pandas DataFrame dễ xử lí hơn
df = credit_data.frame

df.to_csv('Raw_data.csv', index=False)
print("Tạo xong")