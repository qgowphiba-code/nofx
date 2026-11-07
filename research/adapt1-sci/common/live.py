import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from client import Client
ids = [d['domain_id'] if isinstance(d, dict) else d for d in Client().call('GET', '/domains', None, mutation=False).get('domains', [])]
print(len(ids), 'live:', ' '.join(ids))
