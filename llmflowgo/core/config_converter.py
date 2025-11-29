import json
import numpy as np
from typing import Dict, List, Any, Tuple


class ConfigConverter:
    """Configure the converter, responsible for converting user configurations into the standard format required by the algorithm"""
    
    def __init__(self):
        # Predefined network configuration
        self.network_config = {
            "edge_to_edge": {"bandwidth": 100, "power": 0.05},
            "cloud_to_edge": {"bandwidth": 40, "power": 1.5},
            "edge_to_cloud": {"bandwidth": 40, "power": 1.5},
            "device_to_edge": {"bandwidth": 100, "power": 0.05},
            "edge_to_device": {"bandwidth": 100, "power": 0.05},
            "device_to_cloud": {"bandwidth": 40, "power": 1.5},
            "cloud_to_device": {"bandwidth": 40, "power": 1.5},
            "device_to_device": {"bandwidth": 50, "power": 0.02},
            "cloud_to_cloud": {"bandwidth": 200, "power": 2.0}
        }
        
        # Predefined server type configuration
        self.server_types = {
            'cloud': {
                'cloud-small': {'mips': 1600, 'price': 0.96, 'power': 0.15},
                'cloud-medium': {'mips': 3200, 'price': 1.66, 'power': 0.25},
                'cloud-large': {'mips': 4800, 'price': 2.36, 'power': 0.35}
            },
            'edge': {
                'edge-small': {'mips': 1300, 'price': 0.48, 'power': 0.08},
                'edge-medium': {'mips': 2600, 'price': 0.78, 'power': 0.12},
                'edge-large': {'mips': 3900, 'price': 1.08, 'power': 0.18}
            },
            'device': {
                'device-small': {'mips': 1000, 'price': 0, 'power': 0.02},
                'device-medium': {'mips': 2000, 'price': 0, 'power': 0.03},
                'device-large': {'mips': 3000, 'price': 0, 'power': 0.05}
            }
        }

    def convert_dag_config(self, dag_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DAG configuration to standard JSON format
        
        Args:
            dag_config: User configuration of DAG data. Compatible with the following formats:
            - Old version: {node_id: {taskAmount: x, dataAmount: y}, ...}
            - Includes workload and dependencies: {workload: {...}, edges: [...]} or {workload: {...}, predecessors: {...}}
            - Uses task_dependencies mapping: {task_dependencies: {to_id: [from_id, ...]}, ...}
            
        Returns:
            Standard DAG JSON format, containing workload and edges
        """
        if not dag_config:
            # If no configuration is provided, return the default DAG structure
            return self._get_default_dag()
        
        # Identify and extract node configuration source, compatible with multiple input formats
        reserved_keys = {"edges", "predecessors", "task_dependencies"}
        source_nodes: Dict[str, Any] = {}
        if isinstance(dag_config, dict) and isinstance(dag_config.get("workload"), dict):
            source_nodes = dag_config["workload"]
        elif isinstance(dag_config, dict):
            # Old version: Top-level is a node mapping, but exclude reserved keys
            source_nodes = {k: v for k, v in dag_config.items() if k not in reserved_keys}
        
        # Convert node configuration to workload format
        workload: Dict[str, Dict[str, float]] = {}
        for node_id, config in source_nodes.items():
            task_amount = 1.0
            data_amount = 0.0
            # Check the type of config to handle different data formats
            if isinstance(config, dict):
                # If it's a dictionary, extract taskAmount and dataAmount
                task_amount = config.get('taskAmount', 1.0)
                data_amount = config.get('dataAmount', 0.0)
            elif isinstance(config, (int, float)):
                # If it's a number, use it as taskAmount
                task_amount = config
            elif isinstance(config, list) and len(config) > 0:
                # If it's a list, use the first element or default value
                task_amount = config[0] if isinstance(config[0], (int, float)) else 1.0
                if len(config) > 1 and isinstance(config[1], (int, float)):
                    data_amount = config[1]

            workload[str(node_id)] = {'taskAmount': float(task_amount), 'dataAmount': float(data_amount)}
        
        # Prefer using incoming dependency information (edges or predecessors or task_dependencies)
        edges: List[Dict[str, int]] = []
        if isinstance(dag_config, dict) and isinstance(dag_config.get('edges'), list):
            for e in dag_config['edges']:
                if isinstance(e, dict) and 'from' in e and 'to' in e:
                    try:
                        edges.append({"from": int(e['from']), "to": int(e['to'])})
                    except (ValueError, TypeError):
                        continue
        elif isinstance(dag_config, dict) and isinstance(dag_config.get('predecessors'), dict):
            for node, preds in dag_config['predecessors'].items():
                try:
                    node_int = int(node)
                except (ValueError, TypeError):
                    continue
                if isinstance(preds, list):
                    for p in preds:
                        try:
                            edges.append({"from": int(p), "to": node_int})
                        except (ValueError, TypeError):
                            continue
        elif isinstance(dag_config, dict) and isinstance(dag_config.get('task_dependencies'), dict):
            for node, preds in dag_config['task_dependencies'].items():
                try:
                    node_int = int(node)
                except (ValueError, TypeError):
                    continue
                if isinstance(preds, list):
                    for p in preds:
                        try:
                            edges.append({"from": int(p), "to": node_int})
                        except (ValueError, TypeError):
                            continue

        # If dependency information is not provided, a default linear edge will be generated.
        if not edges:
            edges = self._generate_default_edges(len(workload))
        
        return {
            "workload": workload,
            "edges": edges
        }

    def convert_server_config(self, environment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert server configuration to standard JSON format
        
        Args:
            environment_config: Environment configuration, containing cloudSpec, cloudCount, etc.
            
        Returns:
            Standard resources JSON format
        """
        resources = {}
        processor_id = 0
        
        # Process cloud server configuration
        if environment_config.get('cloudSpec') and environment_config.get('cloudCount'):
            cloud_spec = environment_config['cloudSpec']
            cloud_count = environment_config['cloudCount']
            cloud_config = self.server_types['cloud'][cloud_spec]
            
            for i in range(cloud_count):
                resources[str(processor_id)] = {
                    "capacity": cloud_config['mips'] / 100,  # Convert to standard capacity unit
                    "cost_rate": cloud_config['price'],
                    "power": cloud_config['power'],
                    "type": "cloud"
                }
                processor_id += 1
        
        # Process edge server configuration
        if environment_config.get('edgeSpec') and environment_config.get('edgeCount'):
            edge_spec = environment_config['edgeSpec']
            edge_count = environment_config['edgeCount']
            edge_config = self.server_types['edge'][edge_spec]
            
            for i in range(edge_count):
                resources[str(processor_id)] = {
                    "capacity": edge_config['mips'] / 100,
                    "cost_rate": edge_config['price'],
                    "power": edge_config['power'],
                    "type": "edge"
                }
                processor_id += 1
        
        # Process device configuration
        if environment_config.get('deviceSpec') and environment_config.get('deviceCount'):
            device_spec = environment_config['deviceSpec']
            device_count = environment_config['deviceCount']
            device_config = self.server_types['device'][device_spec]
            
            for i in range(device_count):
                resources[str(processor_id)] = {
                    "capacity": device_config['mips'] / 100,
                    "cost_rate": device_config['price'],
                    "power": device_config['power'],
                    "type": "device"
                }
                processor_id += 1
        
        return resources

    def generate_network_config(self, server_count: int, server_types: List[str]) -> Dict[str, Any]:
        """
        Generate network configuration based on server count and types
        
        Args:
            server_count: Total number of servers
            server_types: List of server types, corresponding to each server in order
            
        Returns:
            Network configuration, containing bandwidth and power matrices
        """
        # Initialize matrices   
        bandwidth_matrix = np.zeros((server_count, server_count))
        power_matrix = np.zeros((server_count, server_count))
        
        # Populate matrices
        for i in range(server_count):
            for j in range(server_count):
                if i == j:
                    # Same server, infinite bandwidth, 0 power
                    bandwidth_matrix[i][j] = 1000
                    power_matrix[i][j] = 0.0
                else:
                    # Different servers, determine connection parameters based on types
                    connection_type = self._get_connection_type(server_types[i], server_types[j])
                    config = self.network_config[connection_type]
                    bandwidth_matrix[i][j] = config['bandwidth']
                    power_matrix[i][j] = config['power']
        
        return {
            "bandwidth": bandwidth_matrix.tolist(),
            "link_power_w": power_matrix.tolist()
        }

    def convert_full_config(self, environment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complete environment configuration to standard JSON format
        
        Args:
            environment_config: Complete environment configuration, containing dagConfig and environmentConfig
            
        Returns:
            Dictionary containing all converted data
        """
        dag_config = environment_config.get('dagConfig', {})
        env_config = environment_config.get('environmentConfig', {})
        
        # Convert DAG configuration
        dag_data = self.convert_dag_config(dag_config)
        
        # Convert server configuration
        resources_data = self.convert_server_config(env_config)
        
        # Generate server type list
        server_types = self._extract_server_types(env_config)
        server_count = len(resources_data)
        
        # Generate network configuration
        network_data = self.generate_network_config(server_count, server_types)
        
        return {
            "dag": dag_data,
            "resources": resources_data,
            "network": network_data,
            "metadata": {
                "total_servers": server_count,
                "server_types": server_types,
                "total_tasks": len(dag_data.get('workload', {}))
            }
        }

    def _get_default_dag(self) -> Dict[str, Any]:
        """Return default DAG structure"""
        default_workload = {
            "0": 10, "1": 15, "2": 12, "3": 18, "4": 20,
            "5": 14, "6": 16, "7": 22, "8": 13, "9": 11,
            "10": 19, "11": 17, "12": 21, "13": 15, "14": 18,
            "15": 16, "16": 14, "17": 20, "18": 12, "19": 25
        }
        # Add default dataAmount, for example, half of the task amount
        workload = {k: {'taskAmount': float(v), 'dataAmount': float(v) / 2} for k, v in default_workload.items()}

        return {
            "workload": workload,
            "edges": [
                {"from": 0, "to": 2}, {"from": 0, "to": 3},
                {"from": 1, "to": 4},
                {"from": 2, "to": 5}, {"from": 3, "to": 6},
                {"from": 4, "to": 7},
                {"from": 5, "to": 8}, {"from": 6, "to": 9},
                {"from": 7, "to": 10},
                {"from": 8, "to": 11}, {"from": 9, "to": 12},
                {"from": 10, "to": 13},
                {"from": 11, "to": 14},
                {"from": 12, "to": 15},
                {"from": 13, "to": 16},
                {"from": 14, "to": 17},
                {"from": 15, "to": 18},
                {"from": 16, "to": 19}
            ]
        }

    def _generate_default_edges(self, node_count: int) -> List[Dict[str, int]]:
        """Generate default edges for linear connection"""
        edges = []
        if node_count <= 1:
            return edges
        
        # Simple linear connection
        for i in range(node_count - 1):
            edges.append({"from": i, "to": i + 1})
        
        return edges

    def _get_connection_type(self, type1: str, type2: str) -> str:
        """Determine connection type based on two server types"""
        connection_map = {
            ('cloud', 'cloud'): 'cloud_to_cloud',
            ('cloud', 'edge'): 'cloud_to_edge',
            ('cloud', 'device'): 'cloud_to_device',
            ('edge', 'cloud'): 'edge_to_cloud',
            ('edge', 'edge'): 'edge_to_edge',
            ('edge', 'device'): 'edge_to_device',
            ('device', 'cloud'): 'device_to_cloud',
            ('device', 'edge'): 'device_to_edge',
            ('device', 'device'): 'device_to_device'
        }
        
        return connection_map.get((type1, type2), 'edge_to_edge')

    def _extract_server_types(self, env_config: Dict[str, Any]) -> List[str]:
        """Extract server type list from environment configuration"""
        server_types = []
        
        # Add cloud servers
        cloud_count = env_config.get('cloudCount', 0)
        server_types.extend(['cloud'] * cloud_count)
        
        # Add edge servers  
        edge_count = env_config.get('edgeCount', 0)
        server_types.extend(['edge'] * edge_count)
        
        # Add device servers
        device_count = env_config.get('deviceCount', 0)
        server_types.extend(['device'] * device_count)
        
        return server_types

    def save_converted_data(self, converted_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """
        Save converted data to files in the specified directory.
        
        Args:
            converted_data: The converted data to be saved.
            output_dir: The directory where the files will be saved.
            
        Returns:
            A dictionary mapping file names to their full paths.
        """
        import os
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = {}
        
        # Save DAG data
        dag_path = os.path.join(output_dir, 'dag.json')
        with open(dag_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data['dag'], f, indent=2, ensure_ascii=False)
        file_paths['dag'] = dag_path
        
        # Save resources data
        resources_path = os.path.join(output_dir, 'resources.json')
        with open(resources_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data['resources'], f, indent=2, ensure_ascii=False)
        file_paths['resources'] = resources_path
        
        # Save network data
        network_path = os.path.join(output_dir, 'network.json')
        with open(network_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data['network'], f, indent=2, ensure_ascii=False)
        file_paths['network'] = network_path
        
        return file_paths