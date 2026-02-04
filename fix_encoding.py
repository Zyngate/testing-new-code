with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

with open('config.py', 'w', encoding='ascii', errors='replace') as f:
    f.write(content.encode('ascii', 'replace').decode('ascii'))

print("Fixed encoding")
