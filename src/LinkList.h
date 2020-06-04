#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

#define LENGTH 1.0


//class LinkedList;
class ListNode {
private:
	// _N: size of this level
	int          _N;
	int          _Level;   //start from 0;
	ListNode *   _next;
	ListNode *   _prev;

	double *_U, *_F, *_D;

public:
	ListNode(int n, int l): _N(n),_Level(l),_next(0),_prev(0){
		_U = (double *)malloc(n * n * sizeof(double));
		_F = (double *)malloc(n * n * sizeof(double));
		_D = (double *)malloc(n * n * sizeof(double));

	};
	~ListNode(){
		free(_U);
		free(_F);
		free(_D);
	};
	double* Get_U(){ return _U;};
	double* Get_F(){ return _F;};
	double* Get_D(){ return _D;};
	int Get_N(){ return _N;};
	ListNode* Get_prev(){ return _prev;};
	friend class LinkedList;
};

class LinkedList {
private:
	// first : pointer of the first node.
	ListNode *_first;
	ListNode *_last;
	double _L;
	int _level_N[10]={0};  //maximum level = 10
public:
	LinkedList(int N):_L(LENGTH){
		_first = new ListNode(N, 0);
		_last = _first;
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
	void Push();
	void Pop();
	void Clear();
	int Get_level_now(){return _last->_Level;};
	int Get_N(int level){return _level_N[level];};
	double Get_L(){return _L;};
	ListNode* Get_coarsest_node(){ return _last;};

};


void LinkedList::Push(){
	assert(_level_N[_last->_Level+1]!=0);
	ListNode *newNode = new ListNode(_level_N[_last->_Level+1], _last->_Level+1);
	newNode -> _prev = _last;
	_last -> _next = newNode;
	_last = newNode;
}

void LinkedList::Pop(){
	ListNode *todelete = _last;
	assert(_last->_prev !=0);
	_last = _last -> _prev;
	_last -> _next = 0;
	delete todelete;

}


void LinkedList::Clear() {
	while (_first != 0) {
		ListNode *current = _first;
		_first = _first->_next;
		delete current;  // Delete node as well
	}
}


