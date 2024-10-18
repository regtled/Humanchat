#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  :
# @Time    : 2024/9/11 15:26
# @File    : connection_check.py
# @annotation    : 网络连接检查，测试网络状况，并进行代理设置，防止openai封号

import sys
from pathlib import Path
import random
import requests
import time
import os
# from src.utils.application import Application  # 使用绝对导入
from .log import Log  # 使用绝对导入


os.environ['HTTP_PROXY'] = f'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:7890'

class ConnectionCheck:
    """网络连接检查类，用于测试网络状况并进行代理设置。"""

    @staticmethod
    def get_proxies():
        """获取代理设置。"""
        return {
            "http": os.environ['HTTP_PROXY'],
            "https": os.environ['HTTPS_PROXY'],
        }

    @staticmethod
    def get_ip_data(max_retries=5, initial_delay=2):
        """获取IP数据。

        Args:
            max_retries (int): 最大重试次数。
            initial_delay (int): 初始延迟时间（秒）。

        Returns:
            dict or None: 返回IP数据字典，如果失败则返回None。
        """
        url = 'https://ipapi.co/json/'
        delay = initial_delay
        proxies = ConnectionCheck.get_proxies()
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0'
        }

        for i in range(max_retries):
            try:
                response = requests.request(url=url, proxies=proxies, headers=headers, timeout=10,method="GET")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429:
                    delay_now = random.randint(0, delay)
                    time.sleep(delay_now)
                    delay *= 2
                else:
                    Log.error(f"HTTP error occurred: {http_err}. status_code : {str(response.status_code)}" )
                    break
            except requests.exceptions.RequestException as e:
                Log.error("Error: %s" % e)
                break

        return None

    @staticmethod
    def check_string_in_file(file_path, search_string):
        """检查文件中是否包含指定字符串。

        Args:
            file_path (str): 文件路径。
            search_string (str): 要搜索的字符串。

        Returns:
            bool: 如果找到字符串则返回True，否则返回False。
        """
        try:
            with open(file_path, 'r') as file:
                return any(search_string in line for line in file)
        except FileNotFoundError:
            Log.error("The file %s was not found." % file_path)
            return False
        except Exception as e:
            Log.error("An error occurred: %s" % e)
            return False

    @staticmethod
    def check_traffic():
        """检查网络流量并验证IP地址。

        Returns:
            bool: 如果网络通过测试则返回True，否则返回False。
        """
        data = ConnectionCheck.get_ip_data()
        if data:
            Log.info("IP: %s" % data['ip'])
            Log.info("Country: %s" % data['country_name'])
            log_file_path = Path(__file__).resolve().parent / 'supported_countries.txt'
            if ConnectionCheck.check_string_in_file(log_file_path, data['country_name']):
                Log.info('----------------Internet PASSED the test-------------------')
                return True
            else:
                Log.info('----------------Internet FAILED the test-------------------')
                return False
        else:
            Log.error("Failed to retrieve IP data after multiple attempts")
            Log.info('----------------Internet FAILED the test-------------------')
            return False

# if __name__ == "__main__":
#     res = ConnectionCheck.check_traffic()
#     print(res)
