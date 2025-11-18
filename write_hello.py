import time

# Chờ 3 phút (180 giây)
time.sleep(10)

# Ghi "hello" vào file hello.txt
with open("hello.txt", "w") as f:
    f.write("hello\n")

print("Đã ghi 'hello' vào file hello.txt")
