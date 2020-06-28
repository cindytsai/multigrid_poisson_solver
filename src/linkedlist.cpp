#include <iostream>
#include <stdlib.h>
#include "linkedlist.h"

using namespace std;

ListNode::ListNode(int n){
	N = n;
	U = (double*) malloc(N * N * sizeof(double));
	F = (double*) malloc(N * N * sizeof(double));
	D = (double*) malloc(N * N * sizeof(double));
	nextNode = 0;
	prevNode = 0;
}

LinkedList::~LinkedList(){
	while( firstNode != 0 ){
		ListNode *current = firstNode;
		free(current -> U);
		free(current -> F);
		free(current -> D);
		firstNode = firstNode -> nextNode;
		delete current;
	}
}

void LinkedList::Push_back(int n){
	
	ListNode *newNode = new ListNode(n);

	// There is nothing in the list
	if(firstNode == 0){
		firstNode = newNode;
		lastNode = newNode;
		return;
	}

	// list is not empty
	newNode -> prevNode = lastNode;
	lastNode -> nextNode = newNode;
	lastNode = newNode;
}

void LinkedList::Remove_back(){
	// Free all the allocate memory at the lastNode
	free(lastNode -> U);
	free(lastNode -> F);
	free(lastNode -> D);


	if( firstNode == lastNode && firstNode != 0 ){
		firstNode = 0;
		lastNode = 0;
	}
	else{
		lastNode -> prevNode -> nextNode = 0;
		lastNode = lastNode -> prevNode;
	}
}

void LinkedList::Set_Problem(double l, double o_x, double o_y){
	L = l;
	min_x = o_x;
	min_y = o_y;
}

double LinkedList::Get_L(){
	return L;
}

double* LinkedList::Get_U(){
	return lastNode -> U;
}

double* LinkedList::Get_D(){
	return lastNode -> D;
}

double* LinkedList::Get_F(){
	return lastNode -> F;
}

int LinkedList::Get_N(){
	return lastNode -> N;
}

double* LinkedList::Get_ptr_smoothingError(){
    return &(lastNode -> smoothingError);
}

int LinkedList::Get_prev_N(){
	return (lastNode -> prevNode -> N);
}