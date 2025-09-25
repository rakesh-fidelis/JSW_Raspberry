
"""
Network Information Collection Module
Collects network-related system information
"""

import socket
import platform
import psutil
import subprocess
from typing import Dict, Any, List
from .base_module import BaseModule


class NetworkModule(BaseModule):
    """Network information collection module"""
    
    def __init__(self):
        super().__init__('Network')
        self.is_windows = platform.system().lower() == 'windows'
        self.is_linux = platform.system().lower() == 'linux'
    
    def collect(self) -> Dict[str, Any]:
        """Collect network information"""
        # Note: Network topology scanning is now on-demand only via networkScan command
        # to prevent resource-intensive operations during regular data collection
        network_info = {
            'hostname': self._get_hostname(),
            'interfaces': self._get_network_interfaces(),
            'public_ip': self._get_public_ip(),
            'dns_servers': self._get_dns_servers(),
            'gateway': self._get_default_gateway(),
            'io_counters': self._get_network_io_counters(),
            'wifi_info': self._get_wifi_info(),
            'geolocation': self._get_geolocation()
        }
        
        return network_info
    
    def perform_network_scan(self, subnet: str, scan_type: str = 'ping') -> Dict[str, Any]:
        """Perform on-demand network scanning"""
        self.logger.info(f"Starting network scan for subnet: {subnet}, type: {scan_type}")
        
        scan_results = {
            'subnet': subnet,
            'scan_type': scan_type,
            'discovered_devices': [],
            'local_mac': self._get_local_mac(),
            'scan_timestamp': self._get_current_timestamp()
        }
        
        try:
            if scan_type == 'ping':
                scan_results['discovered_devices'] = self._ping_scan(subnet)
            elif scan_type == 'port':
                scan_results['discovered_devices'] = self._port_scan(subnet)
            elif scan_type == 'full':
                scan_results['discovered_devices'] = self._full_scan(subnet)
            else:
                scan_results['discovered_devices'] = self._ping_scan(subnet)
            
            self.logger.info(f"Network scan completed. Found {len(scan_results['discovered_devices'])} devices")
            
        except Exception as e:
            self.logger.error(f"Error during network scan: {e}")
            scan_results['error'] = str(e)
        
        return scan_results
    
    def _get_hostname(self) -> str:
        """Get system hostname"""
        try:
            return socket.gethostname()
        except Exception as e:
            self.logger.error(f"Error getting hostname: {e}")
            return "unknown"
    
    def _get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get network interface information"""
        interfaces = []
        
        try:
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()
            net_io_counters = psutil.net_io_counters(pernic=True)
        except Exception as e:
            self.logger.error(f"Failed to get network interface data: {e}")
            return interfaces
        
        # Filter out virtual adapters
        virtual_keywords = ['vEthernet', 'VMware', 'Virtual', 'Loopback', 'Hyper-V']
        
        for interface_name, addresses in net_if_addrs.items():
            # Skip virtual adapters by name
            if any(keyword.lower() in interface_name.lower() for keyword in virtual_keywords):
                continue
            
            interface_info = self._get_interface_info(
                interface_name, addresses, net_if_stats, net_io_counters
            )
            
            if interface_info:
                interfaces.append(interface_info)
        
        return interfaces
    
    def _get_interface_info(self, name: str, addresses, stats_dict, io_dict) -> Dict[str, Any]:
        """Get information for a single network interface"""
        interface_info = {
            'name': name,
            'type': self._determine_interface_type(name),
            'addresses': [],
            'ip': None,
            'mac': None,
            'status': 'down',
            'speed': 0,
            'mtu': 0,
            'bytes_sent': 0,
            'bytes_recv': 0,
            'packets_sent': 0,
            'packets_recv': 0
        }
        
        # Get interface statistics
        if name in stats_dict:
            stats = stats_dict[name]
            interface_info['status'] = 'up' if stats.isup else 'down'
            interface_info['speed'] = getattr(stats, 'speed', 0)
            interface_info['mtu'] = getattr(stats, 'mtu', 0)
            
            # Skip interfaces that are down
            if not stats.isup:
                return None
        
        # Get I/O counters
        if name in io_dict:
            io = io_dict[name]
            interface_info['bytes_sent'] = getattr(io, 'bytes_sent', 0)
            interface_info['bytes_recv'] = getattr(io, 'bytes_recv', 0)
            interface_info['packets_sent'] = getattr(io, 'packets_sent', 0)
            interface_info['packets_recv'] = getattr(io, 'packets_recv', 0)
        
        # Process addresses
        for addr in addresses:
            try:
                addr_info = {
                    'family': str(getattr(addr, 'family', 'unknown')),
                    'address': getattr(addr, 'address', ''),
                    'netmask': getattr(addr, 'netmask', None),
                    'broadcast': getattr(addr, 'broadcast', None)
                }
                interface_info['addresses'].append(addr_info)
                
                # Extract primary IP and MAC
                if hasattr(addr, 'family') and addr.family == 2:  # IPv4
                    if not interface_info['ip'] and addr.address != '127.0.0.1':
                        interface_info['ip'] = addr.address
                elif (hasattr(addr, 'address') and addr.address and
                      ':' not in addr.address and len(addr.address) == 17):
                    interface_info['mac'] = addr.address
            except Exception as e:
                self.logger.debug(f"Error processing address for {name}: {e}")
                continue
        
        return interface_info
    
    def _determine_interface_type(self, name: str) -> str:
        """Determine interface type based on name"""
        name_lower = name.lower()
        if 'wi-fi' in name_lower or 'wireless' in name_lower or 'wlan' in name_lower:
            return 'Wi-Fi'
        elif 'ethernet' in name_lower or 'eth' in name_lower or 'en' in name_lower:
            return 'Ethernet'
        elif 'loopback' in name_lower or 'lo' in name_lower:
            return 'Loopback'
        elif 'vpn' in name_lower or 'tap' in name_lower or 'tun' in name_lower:
            return 'VPN'
        else:
            return 'Unknown'
    
    def _get_public_ip(self) -> str:
        """Get public IP address"""
        try:
            import requests
            
            services = [
                'https://api.ipify.org',
                'https://icanhazip.com',
                'https://checkip.amazonaws.com'
            ]
            
            for service in services:
                try:
                    response = requests.get(service, timeout=10)
                    if response.status_code == 200:
                        return response.text.strip()
                except Exception as e:
                    self.logger.debug(f"Failed to get IP from {service}: {e}")
                    continue
        except ImportError:
            self.logger.debug("requests module not available for public IP lookup")
        except Exception as e:
            self.logger.debug(f"Error getting public IP: {e}")
        
        return "unknown"
    
    def _get_dns_servers(self) -> List[str]:
        """Get DNS servers"""
        dns_servers = []
        
        try:
            if self.is_windows:
                dns_servers = self._get_windows_dns_servers()
            elif self.is_linux:
                dns_servers = self._get_linux_dns_servers()
        except Exception as e:
            self.logger.debug(f"Error getting DNS servers: {e}")
        
        return dns_servers
    
    def _get_windows_dns_servers(self) -> List[str]:
        """Get DNS servers on Windows"""
        dns_servers = []
        try:
            result = subprocess.run(['nslookup', 'google.com'],
                                  capture_output=True, text=True, timeout=10)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Server:' in line:
                    dns_ip = line.split(':')[-1].strip()
                    if dns_ip and dns_ip not in dns_servers:
                        dns_servers.append(dns_ip)
        except Exception:
            pass
        return dns_servers
    
    def _get_linux_dns_servers(self) -> List[str]:
        """Get DNS servers on Linux"""
        dns_servers = []
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        dns_ip = line.split()[1]
                        if dns_ip not in dns_servers:
                            dns_servers.append(dns_ip)
        except Exception:
            pass
        return dns_servers
    
    def _get_default_gateway(self) -> str:
        """Get default gateway"""
        try:
            if self.is_windows:
                return self._get_windows_gateway()
            elif self.is_linux:
                return self._get_linux_gateway()
        except Exception as e:
            self.logger.debug(f"Could not determine default gateway: {e}")
        
        return None
    
    def _get_windows_gateway(self) -> str:
        """Get default gateway on Windows"""
        try:
            result = subprocess.run(['route', 'print', '0.0.0.0'],
                                  capture_output=True, text=True, timeout=10)
            for line in result.stdout.split('\n'):
                if '0.0.0.0' in line and 'Gateway' not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
        except Exception:
            pass
        return None
    
    def _get_linux_gateway(self) -> str:
        """Get default gateway on Linux"""
        try:
            result = subprocess.run(['ip', 'route', 'show', 'default'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'default via' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]
        except Exception:
            pass
        return None
    
    def _get_network_io_counters(self) -> Dict[str, Any]:
        """Get network I/O statistics"""
        try:
            io_counters = psutil.net_io_counters()
            return {
                'bytes_sent': io_counters.bytes_sent,
                'bytes_recv': io_counters.bytes_recv,
                'packets_sent': io_counters.packets_sent,
                'packets_recv': io_counters.packets_recv,
                'errin': io_counters.errin,
                'errout': io_counters.errout,
                'dropin': io_counters.dropin,
                'dropout': io_counters.dropout
            }
        except Exception as e:
            self.logger.error(f"Error getting network I/O counters: {e}")
            return {}
    
    def _get_wifi_info(self) -> Dict[str, Any]:
        """Get Wi-Fi information"""
        wifi_info = {'connected': False, 'status': 'unknown'}
        
        try:
            if self.is_windows:
                wifi_info = self._get_windows_wifi_info()
            elif self.is_linux:
                wifi_info = self._get_linux_wifi_info()
        except Exception as e:
            self.logger.debug(f"Error getting Wi-Fi info: {e}")
        
        return wifi_info
    
    def _get_windows_wifi_info(self) -> Dict[str, Any]:
        """Get Wi-Fi information on Windows"""
        try:
            result = subprocess.run(['netsh', 'wlan', 'show', 'interfaces'],
                                  capture_output=True, text=True, timeout=10)
            if 'connected' in result.stdout.lower():
                wifi_info = {'connected': True, 'status': 'connected'}
                # Extract SSID if available
                for line in result.stdout.split('\n'):
                    if 'SSID' in line and ':' in line:
                        wifi_info['ssid'] = line.split(':')[-1].strip()
                        break
                return wifi_info
            else:
                return {'connected': False, 'status': 'disconnected'}
        except Exception:
            return {'connected': False, 'status': 'unknown'}
    
    def _get_linux_wifi_info(self) -> Dict[str, Any]:
        """Get Wi-Fi information on Linux"""
        try:
            result = subprocess.run(['iwconfig'], capture_output=True, text=True, timeout=10)
            if 'ESSID:' in result.stdout:
                wifi_info = {'connected': True, 'status': 'connected'}
                # Extract SSID
                for line in result.stdout.split('\n'):
                    if 'ESSID:' in line:
                        ssid = line.split('ESSID:')[-1].split()[0].strip('"')
                        if ssid != 'off/any':
                            wifi_info['ssid'] = ssid
                        break
                return wifi_info
            else:
                return {'connected': False, 'status': 'disconnected'}
        except Exception:
            return {'connected': False, 'status': 'unknown'}
    
    def _get_geolocation(self) -> Dict[str, Any]:
        """Get geolocation information based on public IP"""
        try:
            import requests
            
            public_ip = self._get_public_ip()
            if public_ip == "unknown":
                return {'location': 'Unknown', 'error': 'Could not determine public IP'}
            
            geo_services = [
                f'https://ipapi.co/{public_ip}/json/',
                f'http://ip-api.com/json/{public_ip}',
                f'https://ipinfo.io/{public_ip}/json'
            ]
            
            for geo_url in geo_services:
                try:
                    geo_response = requests.get(geo_url, timeout=10)
                    if geo_response.status_code == 200:
                        geo_data = geo_response.json()
                        
                        # Extract location information
                        location_parts = []
                        city = geo_data.get('city') or geo_data.get('cityName')
                        region = geo_data.get('region') or geo_data.get('region_name') or geo_data.get('stateProv')
                        country = geo_data.get('country') or geo_data.get('country_name') or geo_data.get('countryName')
                        
                        if city:
                            location_parts.append(city)
                        if region and region != city:
                            location_parts.append(region)
                        if country:
                            location_parts.append(country)
                        
                        return {
                            'location': ', '.join(location_parts) if location_parts else 'Unknown',
                            'isp': geo_data.get('isp') or geo_data.get('org') or geo_data.get('as'),
                            'timezone': geo_data.get('timezone') or geo_data.get('timeZone'),
                            'coordinates': f"{geo_data.get('lat', geo_data.get('latitude', ''))},{geo_data.get('lon', geo_data.get('longitude', ''))}" if geo_data.get('lat') or geo_data.get('latitude') else None
                        }
                except Exception as e:
                    self.logger.debug(f"Failed to get geolocation from {geo_url}: {e}")
                    continue
            
            return {'location': f'IP: {public_ip} (Location lookup failed)'}
            
        except ImportError:
            self.logger.debug("requests module not available for geolocation lookup")
            return {'location': 'Geolocation unavailable (requests module missing)'}
        except Exception as e:
            self.logger.debug(f"Error getting geolocation: {e}")
            return {'location': 'Location detection failed'}

    def _get_local_mac(self) -> str:
        """Get local machine's MAC address"""
        try:
            net_if_addrs = psutil.net_if_addrs()
            for interface_name, addresses in net_if_addrs.items():
                if 'loopback' in interface_name.lower() or 'virtual' in interface_name.lower():
                    continue
                for addr in addresses:
                    if hasattr(addr, 'address') and ':' in addr.address and len(addr.address) == 17:
                        return addr.address
        except Exception:
            pass
        return "Unknown"
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _ping_scan(self, subnet: str) -> List[Dict[str, Any]]:
        """Perform comprehensive ping scan on subnet"""
        discovered_devices = []
        
        try:
            # Parse IP range
            ip_list = self._parse_ip_range(subnet)
            self.logger.info(f"Ping scanning {len(ip_list)} IPs in range: {subnet}")
            
            # Use threading for faster scanning
            import threading
            import concurrent.futures
            
            def scan_ip(ip):
                try:
                    if self._ping_host(ip):
                        device_info = {
                            'ip': ip,
                            'hostname': self._get_hostname_from_ip(ip),
                            'status': 'online',
                            'mac_address': self._get_mac_from_ip(ip),
                            'response_time': self._get_ping_time(ip),
                            'device_type': self._guess_device_type(ip),
                            'os': self._detect_os_simple(ip)
                        }
                        return device_info
                except Exception as e:
                    self.logger.debug(f"Error scanning IP {ip}: {e}")
                return None
            
            # Scan with thread pool for better performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                # Limit to reasonable number for testing
                scan_ips = ip_list[:100] if len(ip_list) > 100 else ip_list
                future_to_ip = {executor.submit(scan_ip, ip): ip for ip in scan_ips}
                
                for future in concurrent.futures.as_completed(future_to_ip):
                    result = future.result()
                    if result:
                        discovered_devices.append(result)
                        self.logger.info(f"Found device: {result['ip']} ({result['hostname']})")
                        
        except Exception as e:
            self.logger.error(f"Error in ping scan: {e}")
        
        self.logger.info(f"Ping scan completed. Found {len(discovered_devices)} devices.")
        return discovered_devices
    
    def _port_scan(self, subnet: str) -> List[Dict[str, Any]]:
        """Perform port scan on subnet"""
        discovered_devices = self._ping_scan(subnet)
        
        # Add port scanning for discovered devices
        for device in discovered_devices:
            device['ports_open'] = self._scan_common_ports(device['ip'])
            
        return discovered_devices
    
    def _full_scan(self, subnet: str) -> List[Dict[str, Any]]:
        """Perform comprehensive scan"""
        discovered_devices = self._port_scan(subnet)
        
        # Add OS detection and service detection
        for device in discovered_devices:
            device['os'] = self._detect_os(device['ip'], device.get('ports_open', []))
            device['services'] = self._detect_services(device['ip'], device.get('ports_open', []))
            
        return discovered_devices
    
    def _parse_ip_range(self, ip_range: str) -> List[str]:
        """Parse different IP range formats"""
        ip_list = []
        
        try:
            if '/' in ip_range:
                # CIDR notation
                import ipaddress
                network = ipaddress.ip_network(ip_range, strict=False)
                ip_list = [str(ip) for ip in network.hosts()]
                
            elif '-' in ip_range:
                # Range notation
                start_ip, end_ip = ip_range.split('-')
                start_parts = list(map(int, start_ip.strip().split('.')))
                end_parts = list(map(int, end_ip.strip().split('.')))
                
                # Simple range implementation for last octet
                if start_parts[:3] == end_parts[:3]:
                    for i in range(start_parts[3], end_parts[3] + 1):
                        ip = f"{start_parts[0]}.{start_parts[1]}.{start_parts[2]}.{i}"
                        ip_list.append(ip)
                        
            elif '*' in ip_range:
                # Wildcard notation
                base = ip_range.replace('*', '')
                if base.endswith('.'):
                    base = base[:-1]
                for i in range(1, 255):
                    ip = f"{base}.{i}"
                    ip_list.append(ip)
                    
        except Exception as e:
            self.logger.error(f"Error parsing IP range {ip_range}: {e}")
            
        return ip_list
    
    def _ping_host(self, ip: str) -> bool:
        """Ping a single host"""
        try:
            if self.is_windows:
                result = subprocess.run(['ping', '-n', '1', '-w', '1000', ip], 
                                      capture_output=True, text=True, timeout=3)
            else:
                result = subprocess.run(['ping', '-c', '1', '-W', '1', ip], 
                                      capture_output=True, text=True, timeout=3)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_hostname_from_ip(self, ip: str) -> str:
        """Get hostname from IP address"""
        try:
            import socket
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except Exception:
            return f"device-{ip.split('.')[-1]}"
    
    def _get_mac_from_ip(self, ip: str) -> str:
        """Get MAC address from IP (ARP table)"""
        try:
            if self.is_windows:
                result = subprocess.run(['arp', '-a', ip], capture_output=True, text=True, timeout=5)
                for line in result.stdout.split('\n'):
                    if ip in line and ('dynamic' in line.lower() or 'static' in line.lower()):
                        parts = line.split()
                        for part in parts:
                            if '-' in part and len(part) == 17:
                                return part.replace('-', ':')
            else:
                result = subprocess.run(['arp', '-n', ip], capture_output=True, text=True, timeout=5)
                for line in result.stdout.split('\n'):
                    if ip in line:
                        parts = line.split()
                        for part in parts:
                            if ':' in part and len(part) == 17:
                                return part
        except Exception:
            pass
        return "Unknown"
    
    def _scan_arp_table(self) -> List[Dict[str, Any]]:
        """Scan complete ARP table for all devices"""
        discovered_devices = []
        
        try:
            if self.is_windows:
                result = subprocess.run(['arp', '-a'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if 'dynamic' in line.lower() or 'static' in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            ip = parts[0]
                            mac = parts[1].replace('-', ':') if '-' in parts[1] else parts[1]
                            if self._is_valid_ip(ip) and self._is_valid_mac(mac):
                                device_info = {
                                    'ip': ip,
                                    'hostname': self._get_hostname_from_ip(ip),
                                    'status': 'online',
                                    'mac_address': mac,
                                    'response_time': 0,
                                    'device_type': self._guess_device_type(ip),
                                    'os': 'Unknown',
                                    'discovery_method': 'ARP Table'
                                }
                                discovered_devices.append(device_info)
            else:
                result = subprocess.run(['arp', '-a'], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split('\n'):
                    if '(' in line and ')' in line and ':' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            ip = parts[1].strip('()')
                            mac = parts[3]
                            if self._is_valid_ip(ip) and self._is_valid_mac(mac):
                                device_info = {
                                    'ip': ip,
                                    'hostname': self._get_hostname_from_ip(ip),
                                    'status': 'online',
                                    'mac_address': mac,
                                    'response_time': 0,
                                    'device_type': self._guess_device_type(ip),
                                    'os': 'Unknown',
                                    'discovery_method': 'ARP Table'
                                }
                                discovered_devices.append(device_info)
                                
        except Exception as e:
            self.logger.error(f"Error scanning ARP table: {e}")
            
        return discovered_devices
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
        except:
            return False
    
    def _is_valid_mac(self, mac: str) -> bool:
        """Validate MAC address format"""
        import re
        return bool(re.match(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$', mac))
    
    def _detect_os_simple(self, ip: str) -> str:
        """Simple OS detection based on ping TTL"""
        try:
            if self.is_windows:
                result = subprocess.run(['ping', '-n', '1', ip], capture_output=True, text=True, timeout=3)
                for line in result.stdout.split('\n'):
                    if 'TTL=' in line:
                        ttl = int(line.split('TTL=')[1].split()[0])
                        if ttl <= 64:
                            return 'Linux/Unix'
                        elif ttl <= 128:
                            return 'Windows'
                        else:
                            return 'Network Device'
            else:
                result = subprocess.run(['ping', '-c', '1', ip], capture_output=True, text=True, timeout=3)
                for line in result.stdout.split('\n'):
                    if 'ttl=' in line:
                        ttl = int(line.split('ttl=')[1].split()[0])
                        if ttl <= 64:
                            return 'Linux/Unix'
                        elif ttl <= 128:
                            return 'Windows'
                        else:
                            return 'Network Device'
        except:
            pass
        return 'Unknown'
    
    def _get_ping_time(self, ip: str) -> int:
        """Get ping response time"""
        try:
            if self.is_windows:
                result = subprocess.run(['ping', '-n', '1', ip], capture_output=True, text=True, timeout=3)
                for line in result.stdout.split('\n'):
                    if 'time=' in line.lower():
                        time_part = line.split('time=')[1].split('ms')[0].strip()
                        return int(float(time_part.replace('<', '')))
            else:
                result = subprocess.run(['ping', '-c', '1', ip], capture_output=True, text=True, timeout=3)
                for line in result.stdout.split('\n'):
                    if 'time=' in line:
                        time_part = line.split('time=')[1].split()[0]
                        return int(float(time_part))
        except Exception:
            pass
        return 0
    
    def _guess_device_type(self, ip: str) -> str:
        """Guess device type based on IP pattern"""
        last_octet = int(ip.split('.')[-1])
        
        if last_octet == 1 or last_octet == 254:
            return 'Router'
        elif 2 <= last_octet <= 10:
            return 'Network Infrastructure'
        elif 100 <= last_octet <= 150:
            return 'Printer'
        elif 200 <= last_octet <= 220:
            return 'IoT Device'
        else:
            return 'Workstation'
    
    def _scan_common_ports(self, ip: str) -> List[int]:
        """Scan common ports"""
        common_ports = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5900]
        open_ports = []
        
        for port in common_ports:
            if self._is_port_open(ip, port):
                open_ports.append(port)
                
        return open_ports
    
    def _is_port_open(self, ip: str, port: int) -> bool:
        """Check if port is open"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _detect_os(self, ip: str, open_ports: List[int]) -> str:
        """Detect OS based on open ports"""
        if 3389 in open_ports:
            return 'Windows'
        elif 22 in open_ports and 80 in open_ports:
            return 'Linux'
        elif 22 in open_ports:
            return 'Unix/Linux'
        elif 5900 in open_ports:
            return 'macOS/Linux'
        else:
            return 'Unknown'
    
    def _detect_services(self, ip: str, open_ports: List[int]) -> List[str]:
        """Detect services based on open ports"""
        services = []
        port_services = {
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            110: 'POP3',
            143: 'IMAP',
            443: 'HTTPS',
            993: 'IMAPS',
            995: 'POP3S',
            3389: 'RDP',
            5900: 'VNC'
        }
        
        for port in open_ports:
            if port in port_services:
                services.append(port_services[port])
                
        return services
