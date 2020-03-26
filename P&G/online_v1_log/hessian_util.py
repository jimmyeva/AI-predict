from pyhessian.client import HessianProxy
from pyhessian import protocol
import json
import datetime

class HessianUtil(object):
    def __init__(self, api_url=None):
        self._api_url = api_url
        self._client = None

    def set_api_url(self, api_url):
        """
        设置hessian的url
        :param api_url: IP + port + service + interface
        :return:
        """
        self._api_url = api_url

    def _init_hessian_client(self):
        """
        初始化hessian客户端
        :return:
        """
        try:
            self._client = HessianProxy(self._api_url, timeout=60000, overload=False)
        except Exception as e:
            print(e)
            raise e

    def request(self, method, *param):
        """
        发送hessian请求
        :param method:
        :param param:
        :return:
        """
        return_result = dict()
        try:
            self._init_hessian_client()
            start_response_time = datetime.datetime.now()
            response = getattr(self._client, method)(*param)
            after_response_time = datetime.datetime.now()
            return_result["time"] = self._time_delta_seconds(start_response_time, after_response_time)
            return_result["code"] = 200
            response = self._deal_response(response)
            if type(response) == str:
                return_result["data"] = response
            else:
                return_result["data"] = response
        except Exception as e:
            after_response_time = datetime.datetime.now()
            return_result["time"] = self._time_delta_seconds(start_response_time, after_response_time)
            print(e)
            return_result["code"] = 500
            return_result["data"] = str(e)
        self._client = None
        return return_result

    def _deal_response(self, response):
        """
        处理返回值
        :param response:
        :return:
        """
        return_result = None
        if self._is_binary(response):
            return_result = response.value.decode("utf-8")
        elif self._is_tuple(response):
            tuple_list = []
            for item in response:
                tuple_list.append(self._deal_response(item))
            return_result = tuple_list
        elif self._is_hessian_object(response):
            object_dict = dict()
            for k, v in response.__dict__.items():
                object_dict[k] = self._deal_response(v)
            return_result = object_dict
        elif self._is_dict(response):
            object_dict = dict()
            for k, v in response.items():
                object_dict[k] = self._deal_response(v)
            return_result = object_dict
        else:
            return_result = response
        return return_result

    def _is_tuple(self, result):
        """
        判断变量是否是元组
        :param result:
        :return:
        """
        if type(result) == tuple:
            return True
        else:
            return False

    def _is_dict(self, result):
        """
        判断变量是否是字典
        :param result:
        :return:
        """
        if type(result) == dict:
            return True
        else:
            return False

    def _is_binary(self, result):
        """
        判断变量是否二进制数据
        :param result:
        :return:
        """
        if type(result) == type(protocol.Binary("")):
            return True
        else:
            return False

    def _is_hessian_object(self, result):
        """
        判断变量是否对象
        :param result:
        :return:
        """
        try:
            res = type(result)._hessian_factory_args
            return True
        except Exception as e:
            return False

    def _time_delta_seconds(self, start_time, end_time):
        """
        计算2个时间的间隔秒数
        :param start_time:
        :param end_time:
        :return:
        """
        interval = end_time - start_time
        seconds = interval.total_seconds()
        return seconds


if __name__ == '__main__':
    hessian_util = HessianUtil("http://192.168.6.24:8080/gateway/router/api?method=biService.getBiDataParam&version=1.0.0&t=134333333&w_appid=supplyChainBI_1.0")
    a = hessian_util.request({"dataSet": "order"})

