from app import app

if __name__ == '__main__':
    port=os.environ.get("PORT")
    app.run(host='0.0.0.0', port=port, debug=True)
