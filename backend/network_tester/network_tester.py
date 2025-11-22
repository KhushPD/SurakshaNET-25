"""
Network Traffic Tester Module
Generates various types of network traffic patterns for testing IDS
"""

import asyncio
import aiohttp
import random
import time
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class TrafficType(Enum):
    """Types of network traffic patterns"""
    NORMAL = "normal"
    HTTP_FLOOD = "http_flood"
    PORT_SCAN = "port_scan"
    SLOWLORIS = "slowloris"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    DDOS = "ddos"

class NetworkTester:
    def __init__(self, target_url: str = "http://localhost:8000"):
        """
        Initialize Network Tester
        
        Args:
            target_url: Target URL for testing (your FastAPI backend)
        """
        self.target_url = target_url
        self.results = []
        self.source_ips = self._generate_source_ips()
        
    def _generate_source_ips(self, count: int = 50) -> List[str]:
        """Generate random source IPs for simulation"""
        ips = []
        for _ in range(count):
            ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}." \
                 f"{random.randint(0, 255)}.{random.randint(1, 254)}"
            ips.append(ip)
        return ips
    
    async def send_request(self, 
                          session: aiohttp.ClientSession,
                          method: str = "GET",
                          endpoint: str = "/",
                          params: Optional[Dict] = None,
                          data: Optional[Dict] = None,
                          headers: Optional[Dict] = None,
                          timeout: int = 5) -> Dict:
        """
        Send a single HTTP request
        
        Returns:
            Dict containing request results
        """
        start_time = time.time()
        result = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "endpoint": endpoint,
            "source_ip": random.choice(self.source_ips),
            "status": None,
            "duration": 0,
            "error": None
        }
        
        try:
            url = f"{self.target_url}{endpoint}"
            
            async with session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                result["status"] = response.status
                result["duration"] = time.time() - start_time
                result["response_size"] = len(await response.text())
                
        except asyncio.TimeoutError:
            result["error"] = "timeout"
            result["duration"] = timeout
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
        
        return result
    
    async def generate_normal_traffic(self, 
                                     requests_count: int = 100,
                                     delay: float = 0.1) -> List[Dict]:
        """
        Generate normal traffic pattern
        
        Args:
            requests_count: Number of requests to send
            delay: Delay between requests (seconds)
        """
        print(f"[NORMAL] Generating {requests_count} normal requests...")
        results = []
        
        endpoints = ["/", "/api/health", "/api/status", "/api/data"]
        
        async with aiohttp.ClientSession() as session:
            for i in range(requests_count):
                endpoint = random.choice(endpoints)
                result = await self.send_request(session, endpoint=endpoint)
                result["traffic_type"] = TrafficType.NORMAL.value
                results.append(result)
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                if (i + 1) % 20 == 0:
                    print(f"  Sent {i + 1}/{requests_count} requests")
        
        print(f"[NORMAL] Completed: {len(results)} requests sent")
        return results
    
    async def generate_http_flood(self, 
                                 requests_count: int = 1000,
                                 concurrent: int = 50) -> List[Dict]:
        """
        Simulate HTTP Flood attack (DDoS)
        High volume of requests in short time
        """
        print(f"[HTTP FLOOD] Generating {requests_count} requests with {concurrent} concurrent...")
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Create batches of concurrent requests
            for batch_start in range(0, requests_count, concurrent):
                batch_size = min(concurrent, requests_count - batch_start)
                
                tasks = [
                    self.send_request(session, endpoint="/api/data")
                    for _ in range(batch_size)
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, dict):
                        result["traffic_type"] = TrafficType.HTTP_FLOOD.value
                        results.append(result)
                
                print(f"  Sent {len(results)}/{requests_count} requests")
        
        print(f"[HTTP FLOOD] Completed: {len(results)} requests sent")
        return results
    
    async def generate_port_scan(self, 
                                ports: Optional[List[int]] = None,
                                target_ip: str = "192.168.1.1") -> List[Dict]:
        """
        Simulate Port Scan attack
        Attempts to connect to multiple ports
        """
        if ports is None:
            ports = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 3306, 3389, 8080, 8443]
        
        print(f"[PORT SCAN] Scanning {len(ports)} ports on {target_ip}...")
        results = []
        
        async with aiohttp.ClientSession() as session:
            for port in ports:
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "traffic_type": TrafficType.PORT_SCAN.value,
                    "source_ip": random.choice(self.source_ips),
                    "destination_ip": target_ip,
                    "destination_port": port,
                    "protocol": "TCP",
                    "status": "scanned"
                }
                results.append(result)
                await asyncio.sleep(0.05)  # Small delay between port attempts
        
        print(f"[PORT SCAN] Completed: {len(results)} ports scanned")
        return results
    
    async def generate_slowloris(self, 
                               connections: int = 20,
                               duration: int = 30) -> List[Dict]:
        """
        Simulate Slowloris attack
        Keep connections open by sending partial requests slowly
        """
        print(f"[SLOWLORIS] Opening {connections} slow connections for {duration}s...")
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(connections):
                task = self._slowloris_connection(session, duration)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if isinstance(r, dict)]
        
        print(f"[SLOWLORIS] Completed: {len(results)} slow connections")
        return results
    
    async def _slowloris_connection(self, 
                                   session: aiohttp.ClientSession, 
                                   duration: int) -> Dict:
        """Single slowloris connection"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "traffic_type": TrafficType.SLOWLORIS.value,
            "source_ip": random.choice(self.source_ips),
            "duration": duration,
            "status": "slow_connection"
        }
        
        try:
            # Keep connection open with slow headers
            headers = {"X-a": "b" * random.randint(1, 10)}
            await self.send_request(session, headers=headers, timeout=duration)
        except:
            pass
        
        return result
    
    async def generate_sql_injection(self, 
                                    attempts: int = 50) -> List[Dict]:
        """
        Simulate SQL Injection attempts
        Send malicious SQL payloads in parameters
        """
        print(f"[SQL INJECTION] Generating {attempts} SQL injection attempts...")
        results = []
        
        sql_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "admin'--",
            "' UNION SELECT NULL--",
            "1'; DROP TABLE users--",
            "' OR 'x'='x",
            "1' AND '1'='1",
        ]
        
        async with aiohttp.ClientSession() as session:
            for i in range(attempts):
                payload = random.choice(sql_payloads)
                params = {"id": payload, "user": payload}
                
                result = await self.send_request(
                    session, 
                    endpoint="/api/data",
                    params=params
                )
                result["traffic_type"] = TrafficType.SQL_INJECTION.value
                result["payload"] = payload
                results.append(result)
                
                await asyncio.sleep(0.1)
        
        print(f"[SQL INJECTION] Completed: {len(results)} attempts")
        return results
    
    async def generate_xss_attack(self, 
                                 attempts: int = 30) -> List[Dict]:
        """
        Simulate XSS (Cross-Site Scripting) attempts
        Send malicious JavaScript payloads
        """
        print(f"[XSS] Generating {attempts} XSS attempts...")
        results = []
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'>",
        ]
        
        async with aiohttp.ClientSession() as session:
            for i in range(attempts):
                payload = random.choice(xss_payloads)
                data = {"comment": payload, "name": payload}
                
                result = await self.send_request(
                    session,
                    method="POST",
                    endpoint="/api/submit",
                    data=data
                )
                result["traffic_type"] = TrafficType.XSS_ATTACK.value
                result["payload"] = payload
                results.append(result)
                
                await asyncio.sleep(0.1)
        
        print(f"[XSS] Completed: {len(results)} attempts")
        return results
    
    async def run_mixed_test(self, 
                           normal_count: int = 100,
                           attack_ratio: float = 0.2) -> Dict:
        """
        Run mixed traffic test with normal and attack patterns
        
        Args:
            normal_count: Number of normal requests
            attack_ratio: Ratio of attack traffic (0.0 to 1.0)
        """
        print(f"\n{'='*60}")
        print(f"MIXED TRAFFIC TEST")
        print(f"Normal: {normal_count} | Attack Ratio: {attack_ratio*100}%")
        print(f"{'='*60}\n")
        
        all_results = []
        
        # Generate normal traffic
        normal_results = await self.generate_normal_traffic(normal_count)
        all_results.extend(normal_results)
        
        # Calculate attack traffic counts
        attack_count = int(normal_count * attack_ratio)
        attacks_per_type = attack_count // 4
        
        # Generate various attacks
        flood_results = await self.generate_http_flood(attacks_per_type, concurrent=20)
        all_results.extend(flood_results)
        
        sql_results = await self.generate_sql_injection(attacks_per_type)
        all_results.extend(sql_results)
        
        xss_results = await self.generate_xss_attack(attacks_per_type // 2)
        all_results.extend(xss_results)
        
        port_results = await self.generate_port_scan()
        all_results.extend(port_results)
        
        # Summary
        summary = {
            "total_requests": len(all_results),
            "normal_traffic": len(normal_results),
            "attack_traffic": len(all_results) - len(normal_results),
            "traffic_breakdown": {},
            "test_duration": sum(r.get("duration", 0) for r in all_results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Count by traffic type
        for result in all_results:
            traffic_type = result.get("traffic_type", "unknown")
            summary["traffic_breakdown"][traffic_type] = \
                summary["traffic_breakdown"].get(traffic_type, 0) + 1
        
        print(f"\n{'='*60}")
        print(f"TEST COMPLETED")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Normal: {summary['normal_traffic']}")
        print(f"Attacks: {summary['attack_traffic']}")
        print(f"{'='*60}\n")
        
        return {
            "summary": summary,
            "results": all_results
        }


# Example usage
async def main():
    # Initialize tester with your FastAPI backend URL
    tester = NetworkTester(target_url="http://localhost:8000")
    
    # Run different test scenarios
    
    # Scenario 1: Normal traffic only
    # await tester.generate_normal_traffic(requests_count=50)
    
    # Scenario 2: HTTP Flood attack
    # await tester.generate_http_flood(requests_count=500, concurrent=50)
    
    # Scenario 3: Port scan
    # await tester.generate_port_scan()
    
    # Scenario 4: SQL Injection attempts
    # await tester.generate_sql_injection(attempts=30)
    
    # Scenario 5: Mixed traffic (recommended for testing IDS)
    results = await tester.run_mixed_test(normal_count=100, attack_ratio=0.3)
    
    print("\nTest Summary:")
    print(f"Total Requests: {results['summary']['total_requests']}")
    print(f"Traffic Breakdown: {results['summary']['traffic_breakdown']}")

if __name__ == "__main__":
    asyncio.run(main())