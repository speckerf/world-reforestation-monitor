from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to your service account key JSON
SERVICE_ACCOUNT_FILE = "auth/gem-eth-analysis-24fe4261f029.json"

def authenticate():
    # Authenticate and build the Drive API service
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    return service

def list_folders_by_name(service, search_term):
    """List all folders in Google Drive that contain a specific substring in their name."""
    query = (
        f"mimeType='application/vnd.google-apps.folder' and name contains '{search_term}'"
    )
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])

    return folders

def list_files(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])
    return files

def delete_folder(service, folder_id):
    """Delete a folder in Google Drive by its ID."""
    try:
        service.files().delete(fileId=folder_id).execute()
        print(f"Folder with ID {folder_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting folder: {e}")


def main():
    service = authenticate()
    search_term = "predictions-mlp_1000m_v01"  # Substring to filter folder names

    folders = list_folders_by_name(service, search_term)
    for folder in folders:
        print(f"Name: {folder['name']}, ID: {folder['id']}")

    # folder_to_delete = [
    #     '1lWvT-rV3xfOJzZAC6466FGQipyko31zI', 
    #     '1Z6kG8hkl_tr6Eugp2BNMg69DBRlCdqth', 
    #     '1RhOF8L-MB-Rhe5vUQjtxE98PCbcN9VzI', 
    #     '12S8jy8nmAU7aJqHzEZ4UFxwRcVu6zVba',
    #     '1ZmKlTIlzExQRTxKa42mPeJsof-tpPiEg', 
    #     '1uAphiAQ0xiyxKEMvZhTeNKs6s2W-gXeb',
    # ]
    # for folder_id in folder_to_delete:
    #     delete_folder(service, folder_id)


    files = list_files(service, folders[0]["id"])
    for file in files:
        print(f"Name: {file['name']}, ID: {file['id']}")

if __name__ == "__main__":
    main()
    