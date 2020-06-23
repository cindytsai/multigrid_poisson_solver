#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

//class LinkedList;
class ListNode {
private:
	// _N: size of this level
	int          _N;
	int          _Level;   //start from 0;
	ListNode *   _next;

	double *_U, *_F, *_D;

public:
	ListNode(int n, int l): _N(n),_Level(l),_next(0){
		_U = (double *)malloc(n * n * sizeof(double));
		_F = (double *)malloc(n * n * sizeof(double));
		_D = (double *)malloc(n * n * sizeof(double));

	};
	~ListNode(){
		free(_U);
		free(_F);
		free(_D)
	};
	double * Get_U{ return _U;};
	double * Get_F{ return _F;};
	double * Get_D{ return _D;};
	friend class LinkedList;
};

class LinkedList {
private:
	// first : pointer of the first node.
	ListNode *_first;
	int _level_N[10]={0};  //maximum level = 10
public:
	LinkedList(int N){
		_first = new ListNode(N, 0);
		_level_N[0]=N;
		for(int i=1; i<10; i++){
			int n_prev = _level_N[i-1];
			if(n_prev<5) break;
			if (n_prev%2 ==0) _level_N[i]=(n_prev+2)/2;
			else _level_N[i]=(n_prev+1)/2;
		}
	};
	~LinkedList(){
		Clear();
	};
	void Push_back(int level);
	void Clear();

};


void LinkedList::Push_back(int level) {
	ListNode *newNode = new ListNode(_level_N[level], level);
	ListNode *current = _first;
	while (current->_next != 0) {
		current = current->_next;
	}
	current->_next = newNode;
}


void LinkedList::Clear() {
	while (_first != 0) {
		ListNode *current = _first;
		_first = _first->_next;
		delete current;  // Delete node as well
	}
}

