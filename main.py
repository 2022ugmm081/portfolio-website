from app import app
import os
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 3000)), debug=False)
