# nacos_client.py
import os

from nacos import NacosClient
import yaml
from sqlalchemy import false


class NacosService:
    def __init__(self, service_name):
        self.client = NacosClient(os.getenv("NACOS_ADDR"), namespace=os.getenv('NACOS_NAMESPACE'))
        self.service_name = service_name
        self.data_id = f"{service_name}.yaml"

    def register_service(self, ip, port, metadata=None):
        """注册服务到 Nacos"""
        self.client.add_naming_instance(
            service_name=self.service_name,
            ip=ip,
            port=port,
            ephemeral=False,
            metadata=metadata
        )
        print(f"Service {self.service_name} registered to Nacos at {ip}:{port}")

    def deregister_service(self, ip, port):
        """从 Nacos 注销服务"""
        self.client.remove_naming_instance(
            service_name=self.service_name,
            ip=ip,
            port=port
        )
        print(f"Service {self.service_name} deregistered from Nacos at {ip}:{port}")

    def get_config(self):
        """从 Nacos 获取配置"""
        config = self.client.get_config(self.data_id, os.getenv('NACOS_NAMESPACE'))
        if config:
            return yaml.safe_load(config)
        return None