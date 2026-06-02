import http.server
import json
import sqlite3
import os
import sys
import urllib.parse
import socketserver

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results'))
DB_PATH = os.path.join(RESULTS_DIR, 'inspection.db')
DASHBOARD_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dashboard'))

def connect_db(path):
    """Establishes database connection configured with WAL and busy timeout for thread safety."""
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as e:
        print(f"Warning: Failed to set PRAGMA parameters: {e}")
    return conn

class InspectionServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Mute standard HTTP logging to keep console clean
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # CORS Headers
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

        if path.startswith("/profiles/"):
            self.serve_profiles(path)
        elif path == "/api/stats":
            self.serve_stats(cors_headers)
        elif path == "/api/history":
            self.serve_history(cors_headers)
        elif path == "/api/pins":
            self.serve_pins(query, cors_headers)
        elif path == "/api/snapshot":
            self.serve_snapshot(query, cors_headers)
        else:
            # Serve static files from dashboard folder
            self.serve_static(path)

    def serve_profiles(self, path):
        profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'profiles'))
        file_name = path.replace("/profiles/", "", 1)
        file_path = os.path.abspath(os.path.join(profiles_dir, file_name))
        
        # Security block to ensure file remains inside profiles directory
        if not file_path.startswith(profiles_dir):
            self.send_response(403)
            self.end_headers()
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def serve_stats(self, cors_headers):
        if not os.path.exists(DB_PATH):
            self.send_json({"error": "Database not found"}, status=404, headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()

            # Overall yield
            cursor.execute("""
                SELECT COUNT(*), 
                       SUM(CASE WHEN overall_status LIKE 'PASS%' THEN 1 ELSE 0 END), 
                       SUM(CASE WHEN overall_status LIKE 'WARN%' THEN 1 ELSE 0 END), 
                       SUM(CASE WHEN overall_status LIKE 'FAIL%' THEN 1 ELSE 0 END) 
                FROM inspection_history
            """)
            total, pass_count, warn_count, fail_count = cursor.fetchone()
            
            # Sum of pin classifications
            cursor.execute("SELECT SUM(passed_pins), SUM(warning_pins), SUM(severe_bent_pins), SUM(missing_pins) FROM inspection_history")
            tot_passed, tot_warn, tot_bent, tot_missing = cursor.fetchone()

            conn.close()

            stats = {
                "total": total or 0,
                "passed": pass_count or 0,
                "warning": warn_count or 0,
                "failed": fail_count or 0,
                "pass_rate_pct": round((pass_count / total * 100.0), 1) if total and total > 0 else 0.0,
                "total_passed_pins": tot_passed or 0,
                "total_warning_pins": tot_warn or 0,
                "total_bent_pins": tot_bent or 0,
                "total_missing_pins": tot_missing or 0
            }
            self.send_json(stats, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_history(self, cors_headers):
        if not os.path.exists(DB_PATH):
            self.send_json([], headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, overall_status, passed_pins, warning_pins, severe_bent_pins, missing_pins, snapshot_file 
                FROM inspection_history 
                ORDER BY timestamp DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            history = []
            for r in rows:
                history.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "overall_status": r[2],
                    "passed_pins": r[3],
                    "warning_pins": r[4],
                    "severe_bent_pins": r[5],
                    "missing_pins": r[6],
                    "snapshot_file": r[7]
                })
            self.send_json(history, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_pins(self, query, cors_headers):
        inspection_id = query.get("id", [None])[0]
        if not inspection_id:
            self.send_json({"error": "Missing 'id' parameter"}, status=400, headers=cors_headers)
            return

        if not os.path.exists(DB_PATH):
            self.send_json([], headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pin_id, status, ncc, shift_px, ratio, detail 
                FROM pin_results 
                WHERE inspection_id = ?
            """, (inspection_id,))
            rows = cursor.fetchall()
            conn.close()

            pins = []
            for r in rows:
                pins.append({
                    "id": r[0],
                    "status": r[1],
                    "ncc": r[2],
                    "shift_px": r[3],
                    "ratio": r[4],
                    "detail": r[5]
                })
            self.send_json(pins, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_snapshot(self, query, cors_headers):
        inspection_id = query.get("id", [None])[0]
        if not inspection_id:
            self.send_response(400)
            self.end_headers()
            return

        if not os.path.exists(DB_PATH):
            self.send_response(404)
            self.end_headers()
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT snapshot_file FROM inspection_history WHERE id = ?", (inspection_id,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0] and os.path.exists(row[0]):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                for k, v in cors_headers.items():
                    self.send_header(k, v)
                self.end_headers()
                with open(row[0], "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

    def serve_static(self, path):
        if path == "/" or path == "":
            file_name = "index.html"
        else:
            file_name = path.lstrip("/")

        file_path = os.path.abspath(os.path.join(DASHBOARD_DIR, file_name))
        
        # Security block to ensure file remains inside dashboard directory
        if not file_path.startswith(os.path.abspath(DASHBOARD_DIR)):
            self.send_response(403)
            self.end_headers()
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            if file_path.endswith(".html"):
                content_type = "text/html"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".json"):
                content_type = "application/json"
            elif file_path.endswith(".png"):
                content_type = "image/png"
            else:
                content_type = "application/octet-stream"

            self.send_header("Content-Type", content_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server that handles concurrent dashboard client requests."""
    daemon_threads = True

def run():
    # Make sure dashboard directory exists
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    
    port = 8000
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, InspectionServerHandler)
    print(f"\n============================================================")
    print(f"      LED PIN INSPECTION HMI OPERATOR DASHBOARD SERVER")
    print(f"============================================================")
    print(f"  Server Running at: http://localhost:{port}")
    print(f"  Query Database:    {DB_PATH}")
    print(f"  Static Directory:  {DASHBOARD_DIR}")
    print(f"  Press Ctrl+C to stop.")
    print(f"============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HMI Dashboard Server...")
        sys.exit(0)

if __name__ == "__main__":
    run()
