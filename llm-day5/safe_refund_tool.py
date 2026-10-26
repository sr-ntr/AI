transactions = {
    "TXN1001": {
        "customer_id": "CUST101",
        "amount": 999.00,
        "currency": "INR",
        "status": "duplicate_verified",
        "refunded_amount": 0.00,
    },
    "TXN1002": {
        "customer_id": "CUST102",
        "amount": 499.00,
        "currency": "INR",
        "status": "completed",
        "refunded_amount": 0.00,
    },
}

refund_operations = {}


def validate_refund_request(
    transaction_id: str,
    requested_amount: float,
    authorized_customer_id: str,
) -> tuple[bool, str]:

    # 1. Verify that the transaction exists.
    transaction = transactions.get(transaction_id)

    if transaction is None:
        return False, "Transaction not found."

    # 2. Verify customer authorization.
    if transaction["customer_id"] != authorized_customer_id:
        return False, "Unauthorized access to this transaction."

    # 3. Verify transaction eligibility.
    if transaction["status"] != "duplicate_verified":
        return False, "Transaction is not eligible for this refund."

    # 4. Validate the requested amount.
    if requested_amount <= 0:
        return False, "Refund amount must be positive."

    refundable_balance = transaction["amount"] - transaction["refunded_amount"]

    if requested_amount > refundable_balance:
        return False, "Requested amount exceeds the refundable balance."

    return True, "Refund request passed validation."


def issue_refund(
    transaction_id: str,
    requested_amount: float,
    authorized_customer_id: str,
    idempotency_key: str,
) -> dict:

    # 1. Validate the idempotency key.
    if not idempotency_key or not idempotency_key.strip():
        return {
            "success": False,
            "message": "A valid idempotency key is required.",
        }

    # 2. Check whether this operation has already been processed.
    if idempotency_key in refund_operations:
        previous = refund_operations[idempotency_key]

        # A key must not be reused for a different operation.
        same_request = (
            previous["transaction_id"] == transaction_id
            and previous["requested_amount"] == requested_amount
            and previous["customer_id"] == authorized_customer_id
        )

        if not same_request:
            return {
                "success": False,
                "message": ("Idempotency key reused with different request details."),
            }

        # Return the previous result instead of issuing another refund.
        return previous["result"]

    # 3. Validate against trusted transaction records.
    valid, message = validate_refund_request(
        transaction_id,
        requested_amount,
        authorized_customer_id,
    )

    if not valid:
        return {
            "success": False,
            "message": message,
        }

    # 4. Simulate a successful refund.
    transaction = transactions[transaction_id]

    transaction["refunded_amount"] += requested_amount

    result = {
        "success": True,
        "message": "Refund processed successfully (simulated).",
        "transaction_id": transaction_id,
        "refunded_amount": requested_amount,
        "currency": transaction["currency"],
    }

    # 5. Record the result against the idempotency key.
    refund_operations[idempotency_key] = {
        "transaction_id": transaction_id,
        "requested_amount": requested_amount,
        "customer_id": authorized_customer_id,
        "result": result,
    }

    return result


if __name__ == "__main__":
    first_result = issue_refund(
        transaction_id="TXN1001",
        requested_amount=999.00,
        authorized_customer_id="CUST101",
        idempotency_key="refund-TXN1001-001",
    )

    print("First request:", first_result)

    second_result = issue_refund(
        transaction_id="TXN1001",
        requested_amount=999.00,
        authorized_customer_id="CUST101",
        idempotency_key="refund-TXN1001-001",
    )

    print("Retry:", second_result)

    third_result = issue_refund(
        transaction_id="TXN1001",
        requested_amount=999.00,
        authorized_customer_id="CUST101",
        idempotency_key="refund-TXN1001-002",
    )

    print("New key:", third_result)
