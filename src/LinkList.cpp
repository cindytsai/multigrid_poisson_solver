#include <iostream>
#include <fstream>
using namespace std;

class LinkedList;

class ListNode {
private:
	// coe : coefficient
	// index : degree of x
	int coe, index;
	ListNode *next;
public:
	ListNode(): coe(0), index(0), next(0) {};
	ListNode(int a, int b): coe(a), index(b), next(0) {};

	friend class LinkedList;
};

class LinkedList {
private:
	// first : pointer of the first node.
	ListNode *first;
public:
	LinkedList(): first(0) {};
	int Len();
	void PrintList(char *argv[]);
	void Push_front(int a, int b);
	void Push_back(int a, int b);
	int Delete(int a);
	void Clear();
	int Get_coe(int p);
	int Get_index(int p);
};

int LinkedList::Len() {
	// How many nodes inside LinkList
	int count = 0;
	ListNode *current = first;
	while (current != 0) {
		count = count + 1;
		current = current->next;
	}
	return count;
}

void LinkedList::PrintList(char *argv[]) {
	ofstream f_write;
	f_write.open(argv[2]);

	// Answer is "0"
	if (first == 0) {
		f_write << "1" << "\n";
		f_write << "0 0" << "\n";
		f_write.close();
		return;
	}

	// Answer other then "0"
	ListNode *current = first;
	// write how many are there.
	f_write << Len() << "\n";

	while (current != 0) {
		f_write << current->coe << " " << current->index << "\n";
		current = current->next;
	}
	f_write.close();
}

void LinkedList::Push_front(int a, int b) {
	ListNode *newNode = new ListNode(a, b);
	newNode->next = first;
	first = newNode;
}

void LinkedList::Push_back(int a, int b) {
	ListNode *newNode = new ListNode(a, b);
	if (first == 0) {
		first = newNode;
		return; // So it won't run the following code.
	}
	ListNode *current = first;
	while (current->next != 0) {
		current = current->next;
	}
	current->next = newNode;
}

int LinkedList::Delete(int a) {
	// delete Node that has 0 coefficient.
	// which is delete a = 0.
	//
	// If there are "NO" a = 0 inside the List,
	// return 1.
	ListNode *current = first, *previous = 0;
	while (current != 0 && current->coe != a) {
		previous = current;
		current = current->next;
	}
	if (current == 0) {
		cout << "No " << a << " inside list.\n";
		return 1;
	}
	else if (current == first) {
		// Delete node at first
		first = current->next;
		delete current;
		// When delete a pointer, point it to NULL, can avoid not necessary bugs.
		current = 0;
		return 0;
	}
	else {
		// Other case, node not at first.
		previous->next = current->next;
		delete current;
		current = 0;
		return 0;
	}
}

void LinkedList::Clear() {
	while (first != 0) {
		ListNode *current = first;
		first = first->next;
		delete current;  // Delete node as well
		current = 0;
	}
}

int LinkedList::Get_coe(int p) {
	// p start from "0"
	// Assume LinkedList is not empty
	ListNode *current = first;
	int p_num = 0;
	while (p_num != p) {
		current = current->next;
		p_num = p_num + 1;
	}
	return current->coe;
}

int LinkedList::Get_index(int p) {
	// p start from "0"
	// Assume LinkedList is not empty
	ListNode *current = first;
	int p_num = 0;
	while (p_num != p) {
		current = current->next;
		p_num = p_num + 1;
	}
	return current->index;
}

LinkedList Addition(LinkedList poly1, LinkedList poly2) {
	cout << "addition\n";
	// Result
	LinkedList poly_result;
	// Length of the LinkedList
	int len1 = poly1.Len() - 1;
	int len2 = poly2.Len() - 1;


	cout << "len1 :" << len1 << "\n";
	cout << "len2 :" << len2 << "\n";
	// current place of the list
	int p_1 = 0;
	int p_2 = 0;
	//temperarily of the coefficient and index
	int temp_coe, temp_index;
	// Compare and intergrate poly1, poly2
	while (len1 != -1 || len2 != -1) {

		if ((len1 != -1) && (len2 != -1)) {
			if (poly1.Get_index(p_1) == poly2.Get_index(p_2)) {
				temp_index = poly1.Get_index(p_1);
				temp_coe = poly1.Get_coe(p_1) + poly2.Get_coe(p_2);
				poly_result.Push_back(temp_coe, temp_index);
				p_1 = p_1 + 1;
				p_2 = p_2 + 1;
				len1 = len1 - 1;
				len2 = len2 - 1;
				continue;
			}
			if (poly1.Get_index(p_1) > poly2.Get_index(p_2)) {
				temp_index = poly1.Get_index(p_1);
				temp_coe = poly1.Get_coe(p_1);
				poly_result.Push_back(temp_coe, temp_index);
				p_1 = p_1 + 1;
				len1 = len1 - 1;
				continue;
			}
			if (poly1.Get_index(p_1) < poly2.Get_index(p_2)) {
				temp_index = poly2.Get_index(p_2);
				temp_coe = poly2.Get_coe(p_2);
				poly_result.Push_back(temp_coe, temp_index);
				p_2 = p_2 + 1;
				len2 = len2 - 1;
				continue;
			}
		}
		if (len2 == -1) {
			temp_index = poly1.Get_index(p_1);
			temp_coe = poly1.Get_coe(p_1);
			poly_result.Push_back(temp_coe, temp_index);
			p_1 = p_1 + 1;
			len1 = len1 - 1;
			continue;
		}
		if (len1 == -1) {
			temp_index = poly2.Get_index(p_2);
			temp_coe = poly2.Get_coe(p_2);
			poly_result.Push_back(temp_coe, temp_index);
			p_2 = p_2 + 1;
			len2 = len2 - 1;
			continue;
		}
	}
	return poly_result;
}

LinkedList Subtraction(LinkedList poly1, LinkedList poly2) {
	cout << "subtraction\n";

	LinkedList poly_result;
	// addition inverse of poly2
	LinkedList poly2_i;
	int len2 = poly2.Len() - 1;
	int p_2 = 0;
	// run through all the node
	while (len2 != -1) {
		poly2_i.Push_back(-1 * poly2.Get_coe(p_2), poly2.Get_index(p_2));
		p_2 = p_2 + 1;
		len2 = len2 - 1;
	}

	poly_result = Addition(poly1, poly2_i);

	return poly_result;
}

LinkedList Multiplication(LinkedList poly1, LinkedList poly2) {
	cout << "multiplication\n";

	LinkedList poly_result;
	LinkedList temp_poly;
	// Length of the LinkedList
	int len1 = poly1.Len() - 1;
	int len2 = poly2.Len() - 1;
	// place of the list
	int p_1 = 0;
	int p_2 = 0;
	// multiply by __
	int m_coe, m_index;

	while (len2 != -1) {
		// multiply these two
		m_coe = poly2.Get_coe(p_2);
		m_index = poly2.Get_index(p_2);

		while (len1 != -1) {

			temp_poly.Push_back(m_coe * poly1.Get_coe(p_1), m_index + poly1.Get_index(p_1));

			p_1 = p_1 + 1;
			len1 = len1 - 1;

		}

		poly_result = Addition(poly_result, temp_poly);

		// go back to the start
		len1 = poly1.Len() - 1;
		p_1 = 0;
		temp_poly.Clear();

		p_2 = p_2 + 1;
		len2 = len2 - 1;
	}

	return poly_result;
}

LinkedList Operation(LinkedList poly1, char sign, LinkedList poly2) {
	// result of the operation
	LinkedList poly_result;

	if (sign == '+') {
		poly_result = Addition(poly1, poly2);
	}

	if (sign == '-') {
		poly_result = Subtraction(poly1, poly2);
	}

	if (sign == '*') {
		poly_result = Multiplication(poly1, poly2);
	}

	// Delete all the 0 coefficient
	while (poly_result.Delete(0) != 1) {
		poly_result.Delete(0);
	}

	return poly_result;
}


int main(int argc, char *argv[]) {

	ifstream f_read;
	f_read.open(argv[1]);

	// num -> how many
	int num;
	LinkedList poly1, poly2;
	int temp_coe, temp_index;
	char ope = '+';  // operator + - *

	// Start with 0, 0
	poly1.Push_back(0, 0);

	while (f_read.eof() != true) {

		f_read >> num;
		cout << "num " << num << "\n";

		while (num != 0) {
			f_read >> temp_coe >> temp_index;
			cout << temp_coe << " " << temp_index << "\n";
			poly2.Push_back(temp_coe, temp_index);
			num = num - 1;
		}

		// former -> poly1; new one -> poly2
		poly1 = Operation(poly1, ope, poly2);
		poly2.Clear();

		if (f_read.eof() == true) {
			break;
		}
		f_read >> ope;
	}

	// close input file, and write in file.
	// poly1 is the answer.
	f_read.close();

	// print out result poly1
	poly1.PrintList(argv);

	return 0;
}