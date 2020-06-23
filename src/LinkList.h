#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

#define LENGTH 1.0


//class LinkedList;
class ListNode {
private:
	// _N: size of this level
	int          N;
	int          Level;   //start from 0;
	ListNode *   next;
	ListNode *   prev;

	double *U, *F, *D;

public:
	ListNode(int n, int l): N(n),Level(l),next(0),prev(0){
		U = (double *)malloc(n * n * sizeof(double));
		F = (double *)malloc(n * n * sizeof(double));
		D = (double *)malloc(n * n * sizeof(double));

	};
	~ListNode(){
		free(U);
		free(F);
		free(D);
	};
	double* Get_U(){ return U;};
	double* Get_F(){ return F;};
	double* Get_D(){ return D;};
	int Get_N(){ return N;};
	ListNode* Get_prev(){ return prev;};
	friend class LinkedList;
};

class LinkedList {
private:
	// first : pointer of the first node.
	ListNode *first;
	ListNode *last;
	double L;
	int level_N[10]={0};  //maximum level = 10
public:
	LinkedList(int N):L(LENGTH){
		first = new ListNode(N, 0);
		last = first;
		level_N[0]=N;
		for(int i=1; i<10; i++){
			int n_prev = level_N[i-1];
			if(n_prev<5) break;
			if (n_prev%2 ==0) level_N[i]=(n_prev+2)/2;
			else level_N[i]=(n_prev+1)/2;
		}
	};
	~LinkedList(){
		Clear();
	};
	void Push();
	void Pop();
	void Clear();
	int Get_level_now(){return last->Level;};
	int Get_N(int level){return level_N[level];};
	double Get_L(){return L;};
	ListNode* Get_coarsest_node(){ return last;};

};


void LinkedList::Push(){
	assert(level_N[last->Level+1]!=0);
	ListNode *newNode = new ListNode(level_N[last->Level+1], last->Level+1);
	newNode -> prev = last;
	last -> next = newNode;
	last = newNode;
}

void LinkedList::Pop(){
	ListNode *todelete = last;
	assert(last->prev !=0);
	last = last -> prev;
	last -> next = 0;
	delete todelete;

}


void LinkedList::Clear() {
	while (first != 0) {
		ListNode *current = first;
		first = first->next;
		delete current;  // Delete node as well
	}
}


