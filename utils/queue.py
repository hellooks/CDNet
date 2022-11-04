import torch
class ft_Queue:
    class _Node :
        def __init__(self, element, next):
            self._element = element
            self._next = next

        def get_elements(self):
            return self._element

        def set_elements(self, num):
            self._element = num        

    def __init__(self) :
        self._head = None
        self._tail = None
        self._size = 0

    def targetft(self):  
        tmp = self._head
        step=1
        while tmp != None :
            if step==1:
                cat_tensor = tmp.get_elements()
                step=0
            else:
                cat_tensor=torch.cat([tmp.get_elements(),cat_tensor],dim=0)
            # print(cat_tensor)
            tmp = tmp._next     
        return cat_tensor   

    def __len__(self) :
        return self._size

    def is_empty(self) :
        return self._size == 0

    def first(self) :
        if self.is_empty() :
            raise Empty('Queue is empty')
        return self._head._element

    def dequeue(self) :
        if self.is_empty():
            raise Empty('Queue is empty')
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        if self.is_empty() :
            self._tail = None
        return answer

    def enqueue(self, e) :
        newest = self._Node(e,None)
        if self.is_empty() :
            self._head = newest
        else :
            self._tail._next = newest
        self._tail = newest
        self._size += 1
class Empty(Exception) :
    pass