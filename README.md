# **Dashboard Documentation**
```
Tên: Trần Khánh Linh
Mã sinh viên: B21DCCN074
Lớp: E21TTNT
```

## **Step to run**
- Download code folder
- Open code in IDLE (recommend VSCode)
- Make sure you are inside `denmark/` 
- Run the following commands to run the server up

```bash
cd /path/to/denmark 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
``` 

- Now you can see in the terminal

```bash
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: on
```

- You can now access the dashboard in two ways:
  - Open [http://127.0.0.1:8050/](http://127.0.0.1:8050/) 
  - or through `index.html`: Right-click on `index.html`, copy the path, and paste it into any browser

## **Available features**
- Automatic updates every 2 seconds
- Year slider to view graph of each year
- Interactive: Please try click on points, zoom in certain part, spikes
