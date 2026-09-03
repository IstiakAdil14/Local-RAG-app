from app.services.vector_store.service import QdrantService
try:
    svc = QdrantService()
    print("Init passed")
    status = svc.get_status()
    print(status)
except Exception as e:
    import traceback
    traceback.print_exc()
